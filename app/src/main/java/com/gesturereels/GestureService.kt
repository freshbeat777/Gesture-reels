package com.gesturereels

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.abs

class GestureService : LifecycleService() {

    companion object {
        private const val TAG = "GestureService"
        // --- Tuning knobs ---
        private const val WINDOW_MS = 400L         // time window to measure a nod/turn
        private const val PITCH_THRESH = 0.12f      // how far the nose must move vertically (relative to face height) to count as a nod
        private const val YAW_THRESH = 0.12f        // how far the nose must move horizontally (relative to face width) to count as a turn
        private const val COOLDOWN_MS = 400L        // minimum pause between triggers
        private const val REARM_DELTA = 0.06f       // must return this far back toward center before the next trigger is allowed
        private const val REARM_TIMEOUT_MS = 1500L  // ...or this much time must pass
        private const val DEBUG = true              // show small status messages (set false to hide)
        // If "turn right" fires on a LEFT turn instead (or vice versa), flip this to true.
        private const val MIRROR_YAW = false
    }

    private var landmarker: FaceLandmarker? = null
    private val executor = Executors.newSingleThreadExecutor()

    // Pitch (nod up / nod down) state
    private val pitchHistory = ArrayDeque<Pair<Long, Float>>()
    private var pitchLastTrigger = 0L
    private var pitchArmed = true
    private var pitchExtreme = 0f
    private var pitchLastWasDown = false

    // Yaw (turn right -> back) state
    private val yawHistory = ArrayDeque<Pair<Long, Float>>()
    private var yawLastTrigger = 0L
    private var yawArmed = true
    private var yawExtreme = 0f

    private var lastDebug = 0L
    private val ui = Handler(Looper.getMainLooper())
    private var toast: Toast? = null
    private var faceSeen = false

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
        setupLandmarker()
        startCamera()
    }

    private fun say(msg: String) {
        ui.post {
            toast?.cancel()
            toast = Toast.makeText(applicationContext, msg, Toast.LENGTH_SHORT).also { it.show() }
        }
    }

    private fun startForegroundNotification() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel("gesture", "Gesture control", NotificationManager.IMPORTANCE_LOW)
        )
        val n = Notification.Builder(this, "gesture")
            .setContentTitle("Gesture control is running")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
        } else {
            startForeground(1, n)
        }
    }

    private fun setupLandmarker() {
        val base = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
        val opts = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.3f)
            .setMinFacePresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setResultListener { result, _ -> onResult(result) }
            .setErrorListener { e ->
                Log.e(TAG, "MediaPipe error", e)
                say("MediaPipe error: " + e.message)
            }
            .build()
        try {
            landmarker = FaceLandmarker.createFromOptions(this, opts)
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
            say("Model load failed: " + e.message)
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { proxy -> analyze(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val lm = landmarker ?: return
            val bmp = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            val now = SystemClock.uptimeMillis()
            lm.detectAsync(BitmapImageBuilder(rotated).build(), now)
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
            say("analyze error: " + e.message)
        } finally {
            proxy.close()
        }
    }

    private fun onResult(result: FaceLandmarkerResult) {
        val face = result.faceLandmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (face == null) {
            if (faceSeen && DEBUG) say("Face lost")
            faceSeen = false
            pitchHistory.clear()
            yawHistory.clear()
            return
        }
        if (!faceSeen && DEBUG) say("Face detected")
        faceSeen = true

        // 1 = nose tip, 10 = forehead, 152 = chin, 234 = right cheek, 454 = left cheek
        val nose = face[1]
        val forehead = face[10]
        val chin = face[152]
        val rightCheek = face[234]
        val leftCheek = face[454]

        val faceH = abs(chin.y() - forehead.y())
        val faceW = abs(leftCheek.x() - rightCheek.x())
        if (faceH < 0.01f || faceW < 0.01f) return

        // Nose's vertical position relative to the face's vertical center, normalized by face height
        val pitchD = (nose.y() - (forehead.y() + chin.y()) / 2f) / faceH
        pitchHistory.addLast(now to pitchD)
        while (pitchHistory.isNotEmpty() && now - pitchHistory.first().first > WINDOW_MS) pitchHistory.removeFirst()
        val pitchDelta = pitchD - pitchHistory.first().second

        // Nose's horizontal position relative to the face's horizontal center, normalized by face width
        var yawD = (nose.x() - (leftCheek.x() + rightCheek.x()) / 2f) / faceW
        if (MIRROR_YAW) yawD = -yawD
        yawHistory.addLast(now to yawD)
        while (yawHistory.isNotEmpty() && now - yawHistory.first().first > WINDOW_MS) yawHistory.removeFirst()
        val yawDelta = yawD - yawHistory.first().second

        if (DEBUG && now - lastDebug > 2500) {
            say("pitch=%.2f yaw=%.2f".format(pitchD, yawD))
            lastDebug = now
        }

        handlePitch(pitchD, pitchDelta, now)
        handleYaw(yawD, yawDelta, now)
    }

    private fun handlePitch(d: Float, delta: Float, now: Long) {
        if (!pitchArmed) {
            if (pitchLastWasDown) {
                // last trigger was a nod DOWN; wait for it to return back up toward center
                if (d > pitchExtreme) pitchExtreme = d
                if (pitchExtreme - d > REARM_DELTA || now - pitchLastTrigger > REARM_TIMEOUT_MS) {
                    pitchArmed = true
                    pitchHistory.clear()
                }
            } else {
                // last trigger was a nod UP; wait for it to return back down toward center
                if (d < pitchExtreme) pitchExtreme = d
                if (d - pitchExtreme > REARM_DELTA || now - pitchLastTrigger > REARM_TIMEOUT_MS) {
                    pitchArmed = true
                    pitchHistory.clear()
                }
            }
            return
        }
        if (now - pitchLastTrigger < COOLDOWN_MS || pitchHistory.size < 2) return
        if (delta < -PITCH_THRESH) firePitch(true, now, d)        // nod up -> next reel
        else if (delta > PITCH_THRESH) firePitch(false, now, d)   // nod down -> previous reel
    }

    private fun firePitch(next: Boolean, now: Long, d: Float) {
        pitchLastTrigger = now
        pitchArmed = false
        pitchLastWasDown = !next
        pitchExtreme = d
        pitchHistory.clear()
        if (DEBUG) say(if (next) "NEXT (nod up)" else "PREVIOUS (nod down)")
        SwipeAccessibilityService.swipe(next)
    }

    private fun handleYaw(d: Float, delta: Float, now: Long) {
        if (!yawArmed) {
            // last trigger was a turn RIGHT; wait for it to return back toward center
            if (d > yawExtreme) yawExtreme = d
            if (yawExtreme - d > REARM_DELTA || now - yawLastTrigger > REARM_TIMEOUT_MS) {
                yawArmed = true
                yawHistory.clear()
            }
            return
        }
        if (now - yawLastTrigger < COOLDOWN_MS || yawHistory.size < 2) return
        if (delta > YAW_THRESH) fireYaw(now, d)   // turn right -> back
    }

    private fun fireYaw(now: Long, d: Float) {
        yawLastTrigger = now
        yawArmed = false
        yawExtreme = d
        yawHistory.clear()
        if (DEBUG) say("BACK (turn right)")
        SwipeAccessibilityService.goBack()
    }

    override fun onDestroy() {
        landmarker?.close()
        executor.shutdown()
        super.onDestroy()
    }
}
