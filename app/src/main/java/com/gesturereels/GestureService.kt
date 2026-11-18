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
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.hypot

class GestureService : LifecycleService() {

    companion object {
        private const val TAG = "GestureService"
        // --- Tuning knobs ---
        private const val WINDOW_MS = 400L      // time window to measure a flick
        private const val THRESH = 0.35f        // how far the fingertip must move (in palm lengths)
        private const val COOLDOWN_MS = 400L    // minimum pause between swipes
        private const val REARM_DELTA = 0.3f      // finger must curl back by this much to allow the next swipe
        private const val REARM_TIMEOUT_MS = 1500L // ...or this much time must pass
        private const val DEBUG = true            // show small status messages (set false to hide)
    }

    private var landmarker: HandLandmarker? = null
    private val executor = Executors.newSingleThreadExecutor()
    private val history = ArrayDeque<Pair<Long, Float>>()
    private var lastTrigger = 0L
    private var armed = true
    private var extremeD = 0f
    private var lastWasDown = false
    private var lastDebug = 0L
    private val ui = Handler(Looper.getMainLooper())
    private var toast: Toast? = null
    private var handSeen = false
    private var lastReport = 0L
    @Volatile private var maxMove = 0f
    @Volatile private var frames = 0
    @Volatile private var results = 0
    @Volatile private var imgW = 1f
    @Volatile private var imgH = 1f
    @Volatile private var rotation = 0

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
        val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        val opts = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setResultListener { result, _ -> onResult(result) }
            .setErrorListener { e ->
                Log.e(TAG, "MediaPipe error", e)
                say("MediaPipe error: " + e.message)
            }
            .build()
        try {
            landmarker = HandLandmarker.createFromOptions(this, opts)
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
            rotation = proxy.imageInfo.rotationDegrees
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            imgW = rotated.width.toFloat()
            imgH = rotated.height.toFloat()
            frames++
            val now = SystemClock.uptimeMillis()

            lm.detectAsync(BitmapImageBuilder(rotated).build(), now)
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
            say("analyze error: " + e.message)
        } finally {
            proxy.close()
        }
    }

    private fun onResult(result: HandLandmarkerResult) {
        results++
        val hand = result.landmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (hand == null) {
            if (handSeen && DEBUG) say("Hand lost")
            handSeen = false
            history.clear()
            return
        }
        if (!handSeen && DEBUG) say("Hand detected")
        handSeen = true
        val w = imgW
        val h = imgH
        // 0 = wrist, 9 = middle knuckle, 5 = index knuckle, 8 = index fingertip
        val palm = hypot(
            (hand[9].x() - hand[0].x()) * w,
            (hand[9].y() - hand[0].y()) * h
        )
        if (palm < 1f) return

        // How far the fingertip is above(-) / below(+) its knuckle, in palm lengths
        val d = (hand[8].y() - hand[5].y()) * h / palm

        history.addLast(now to d)
        while (history.isNotEmpty() && now - history.first().first > WINDOW_MS) history.removeFirst()
        val delta = d - history.first().second
        if (abs(delta) > maxMove) maxMove = abs(delta)

        // After a flick, wait for the finger to return from that same direction before allowing another trigger.
        if (DEBUG && now - lastDebug > 2500) {
            say("d=%.2f move=%.2f armed=%b".format(d, maxMove, armed))
            maxMove = 0f
            lastDebug = now
        }
        if (!armed) {
            if (lastWasDown) {
                // last trigger was a DOWN flick; wait for the finger to curl back UP
                if (d > extremeD) extremeD = d
                if (extremeD - d > REARM_DELTA || now - lastTrigger > REARM_TIMEOUT_MS) {
                    armed = true
                    history.clear()
                }
            } else {
                // last trigger was an UP flick; wait for the finger to curl back DOWN
                if (d < extremeD) extremeD = d
                if (d - extremeD > REARM_DELTA || now - lastTrigger > REARM_TIMEOUT_MS) {
                    armed = true
                    history.clear()
                }
            }
            return
        }
        if (now - lastTrigger < COOLDOWN_MS || history.size < 2) return
        if (delta < -THRESH) fire(true, now, d)         // finger flicked up -> next reel
        else if (delta > THRESH) fire(false, now, d)    // finger flicked down -> previous reel
    }

    private fun fire(next: Boolean, now: Long, d: Float) {
        lastTrigger = now
        armed = false
        lastWasDown = !next
        extremeD = d
        history.clear()
        if (DEBUG) say(if (next) "NEXT" else "PREVIOUS")
        SwipeAccessibilityService.swipe(next)
    }

    override fun onDestroy() {
        landmarker?.close()
        executor.shutdown()
        super.onDestroy()
    }
}
