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
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.hypot

class GestureService : LifecycleService() {

    companion object {
        private const val TAG = "GestureService"

        // --- Head (face) tuning knobs ---
        private const val WINDOW_MS = 400L
        private const val PITCH_THRESH = 0.12f
        private const val YAW_THRESH = 0.12f
        private const val COOLDOWN_MS = 400L
        private const val REARM_DELTA = 0.06f
        private const val REARM_TIMEOUT_MS = 1500L
        // If "turn right" fires on a LEFT turn instead (or vice versa), flip this to true.
        private const val MIRROR_YAW = false
        // Head gestures only fire when both eyes are at least this "closed" (0 = open, 1 = fully closed)
        private const val EYE_CLOSED_THRESH = 0.5f

        // --- Finger tuning knobs ---
        private const val FINGER_WINDOW_MS = 400L
        private const val FINGER_THRESH = 0.35f
        private const val FINGER_COOLDOWN_MS = 400L
        private const val FINGER_REARM_DELTA = 0.3f
        private const val FINGER_REARM_TIMEOUT_MS = 1500L

        private const val DEBUG = true // show small status messages (set false to hide)
    }

    private var faceLandmarker: FaceLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private val executor = Executors.newSingleThreadExecutor()

    // Shared per-frame size, set in analyze(), read by both result callbacks
    @Volatile private var imgW = 1f
    @Volatile private var imgH = 1f

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

    // Finger (flick up -> next only) state
    private val fingerHistory = ArrayDeque<Pair<Long, Float>>()
    private var fingerLastTrigger = 0L
    private var fingerArmed = true
    private var fingerMinD = 10f
    private var fingerSeen = false

    private var lastDebug = 0L
    private val ui = Handler(Looper.getMainLooper())
    private var toast: Toast? = null
    private var faceSeen = false

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
        setupLandmarkers()
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

    private fun setupLandmarkers() {
        try {
            val faceBase = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
            val faceOpts = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(faceBase)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.3f)
                .setMinFacePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setOutputFaceBlendshapes(true)
                .setResultListener { result, _ -> onFaceResult(result) }
                .setErrorListener { e ->
                    Log.e(TAG, "Face MediaPipe error", e)
                    say("Face error: " + e.message)
                }
                .build()
            faceLandmarker = FaceLandmarker.createFromOptions(this, faceOpts)
        } catch (e: Exception) {
            Log.e(TAG, "Face model load failed", e)
            say("Face model load failed: " + e.message)
        }

        try {
            val handBase = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            val handOpts = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(handBase)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setResultListener { result, _ -> onHandResult(result) }
                .setErrorListener { e ->
                    Log.e(TAG, "Hand MediaPipe error", e)
                    say("Hand error: " + e.message)
                }
                .build()
            handLandmarker = HandLandmarker.createFromOptions(this, handOpts)
        } catch (e: Exception) {
            Log.e(TAG, "Hand model load failed", e)
            say("Hand model load failed: " + e.message)
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
            val fl = faceLandmarker
            val hl = handLandmarker
            if (fl == null && hl == null) return
            val bmp = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            imgW = rotated.width.toFloat()
            imgH = rotated.height.toFloat()
            val now = SystemClock.uptimeMillis()
            val mpImage = BitmapImageBuilder(rotated).build()
            fl?.detectAsync(mpImage, now)
            hl?.detectAsync(mpImage, now)
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
            say("analyze error: " + e.message)
        } finally {
            proxy.close()
        }
    }

    // ===================== HEAD (face) =====================

    private fun onFaceResult(result: FaceLandmarkerResult) {
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

        val pitchD = (nose.y() - (forehead.y() + chin.y()) / 2f) / faceH
        pitchHistory.addLast(now to pitchD)
        while (pitchHistory.isNotEmpty() && now - pitchHistory.first().first > WINDOW_MS) pitchHistory.removeFirst()
        val pitchDelta = pitchD - pitchHistory.first().second

        var yawD = (nose.x() - (leftCheek.x() + rightCheek.x()) / 2f) / faceW
        if (MIRROR_YAW) yawD = -yawD
        yawHistory.addLast(now to yawD)
        while (yawHistory.isNotEmpty() && now - yawHistory.first().first > WINDOW_MS) yawHistory.removeFirst()
        val yawDelta = yawD - yawHistory.first().second

        var eyesClosed = false
        val blendshapesList = result.faceBlendshapes()
        if (blendshapesList.isPresent && blendshapesList.get().isNotEmpty()) {
            var blinkL = 0f
            var blinkR = 0f
            for (c in blendshapesList.get()[0]) {
                when (c.categoryName()) {
                    "eyeBlinkLeft" -> blinkL = c.score()
                    "eyeBlinkRight" -> blinkR = c.score()
                }
            }
            eyesClosed = (blinkL + blinkR) / 2f > EYE_CLOSED_THRESH
        }

        if (DEBUG && now - lastDebug > 2500) {
            say("pitch=%.2f yaw=%.2f eyes=%s".format(pitchD, yawD, if (eyesClosed) "closed" else "open"))
            lastDebug = now
        }

        handlePitch(pitchD, pitchDelta, now, eyesClosed)
        handleYaw(yawD, yawDelta, now, eyesClosed)
    }

    private fun handlePitch(d: Float, delta: Float, now: Long, eyesClosed: Boolean) {
        if (!pitchArmed) {
            if (pitchLastWasDown) {
                if (d > pitchExtreme) pitchExtreme = d
                if (pitchExtreme - d > REARM_DELTA || now - pitchLastTrigger > REARM_TIMEOUT_MS) {
                    pitchArmed = true
                    pitchHistory.clear()
                }
            } else {
                if (d < pitchExtreme) pitchExtreme = d
                if (d - pitchExtreme > REARM_DELTA || now - pitchLastTrigger > REARM_TIMEOUT_MS) {
                    pitchArmed = true
                    pitchHistory.clear()
                }
            }
            return
        }
        if (now - pitchLastTrigger < COOLDOWN_MS || pitchHistory.size < 2 || !eyesClosed) return
        if (delta < -PITCH_THRESH) firePitch(true, now, d)
        else if (delta > PITCH_THRESH) firePitch(false, now, d)
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

    private fun handleYaw(d: Float, delta: Float, now: Long, eyesClosed: Boolean) {
        if (!yawArmed) {
            if (d > yawExtreme) yawExtreme = d
            if (yawExtreme - d > REARM_DELTA || now - yawLastTrigger > REARM_TIMEOUT_MS) {
                yawArmed = true
                yawHistory.clear()
            }
            return
        }
        if (now - yawLastTrigger < COOLDOWN_MS || yawHistory.size < 2 || !eyesClosed) return
        if (delta > YAW_THRESH) fireYaw(now, d)
    }

    private fun fireYaw(now: Long, d: Float) {
        yawLastTrigger = now
        yawArmed = false
        yawExtreme = d
        yawHistory.clear()
        if (DEBUG) say("BACK (turn right)")
        SwipeAccessibilityService.goBack()
    }

    // ===================== FINGER (flick up only) =====================

    private fun onHandResult(result: HandLandmarkerResult) {
        val hand = result.landmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (hand == null) {
            if (fingerSeen && DEBUG) say("Hand lost")
            fingerSeen = false
            fingerHistory.clear()
            return
        }
        if (!fingerSeen && DEBUG) say("Hand detected")
        fingerSeen = true

        // 0 = wrist, 9 = middle knuckle, 5 = index knuckle, 8 = index fingertip
        val palm = hypot(
            (hand[9].x() - hand[0].x()) * imgW,
            (hand[9].y() - hand[0].y()) * imgH
        )
        if (palm < 1f) return

        val d = (hand[8].y() - hand[5].y()) * imgH / palm
        fingerHistory.addLast(now to d)
        while (fingerHistory.isNotEmpty() && now - fingerHistory.first().first > FINGER_WINDOW_MS) fingerHistory.removeFirst()
        val delta = d - fingerHistory.first().second

        if (!fingerArmed) {
            if (d < fingerMinD) fingerMinD = d
            if (d - fingerMinD > FINGER_REARM_DELTA || now - fingerLastTrigger > FINGER_REARM_TIMEOUT_MS) {
                fingerArmed = true
                fingerHistory.clear()
            }
            return
        }
        if (now - fingerLastTrigger < FINGER_COOLDOWN_MS || fingerHistory.size < 2) return
        if (delta < -FINGER_THRESH) fireFinger(now)   // finger flicked up -> next reel
    }

    private fun fireFinger(now: Long) {
        fingerLastTrigger = now
        fingerArmed = false
        fingerMinD = 10f
        fingerHistory.clear()
        if (DEBUG) say("NEXT (finger up)")
        SwipeAccessibilityService.swipe(true)
    }

    override fun onDestroy() {
        faceLandmarker?.close()
        handLandmarker?.close()
        executor.shutdown()
        super.onDestroy()
    }
}
