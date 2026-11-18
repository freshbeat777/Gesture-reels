package com.gesturereels

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView
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
import kotlin.math.hypot

class GestureService : LifecycleService() {

    companion object {
        private const val TAG = "GestureService"

        // --- Finger flick (up -> next) tuning knobs ---
        private const val FINGER_WINDOW_MS = 400L
        private const val FINGER_THRESH = 0.35f
        private const val FINGER_COOLDOWN_MS = 400L
        private const val FINGER_REARM_DELTA = 0.3f
        private const val FINGER_REARM_TIMEOUT_MS = 1500L

        // --- Palm pose (open -> previous) tuning knobs ---
        private const val PALM_COOLDOWN_MS = 500L
        // A finger counts as "extended" when its tip sits at least this many times
        // farther from the wrist than its middle (PIP) joint.
        private const val EXTENDED_RATIO = 1.0f
        private const val OPEN_FINGER_COUNT = 3   // this many extended fingers (of 4) = palm open
        private const val CLOSED_FINGER_COUNT = 1 // this many or fewer extended = fist / closed, resets the arming

        // --- Eyes closed -> toggle cursor ---
        private const val EYE_CLOSED_THRESH = 0.5f   // 0 = open, 1 = fully closed (MediaPipe blink score)
        private const val EYES_CLOSED_HOLD_MS = 4000L // how long eyes must stay closed to toggle the cursor

        // --- Double blink -> click ---
        private const val BLINK_MIN_MS = 80L    // ignore flicker shorter than this
        private const val BLINK_MAX_MS = 500L   // a closure longer than this is a hold, not a blink
        private const val DOUBLE_BLINK_WINDOW_MS = 700L // two blinks within this window = a click

        // --- Head movement -> cursor movement ---
        // How far the cursor moves (as a fraction of screen size) per unit of head movement.
        // Higher = more sensitive / cursor moves farther for the same head motion.
        private const val CURSOR_SENSITIVITY = 3.0f
        // If moving your head left moves the cursor right (or vice versa), flip this to true.
        private const val MIRROR_CURSOR_X = false

        private const val DEBUG = true // show small status messages (set false to hide)
    }

    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile private var imgW = 1f
    @Volatile private var imgH = 1f

    private var handSeen = false

    // Eyes-closed-hold / cursor show-hide state
    private var eyesClosedSince = 0L
    private var closureHandled = false
    private var cursorView: TextView? = null
    private var windowManager: WindowManager? = null

    // Double-blink click state
    private val blinkTimes = ArrayDeque<Long>()

    // Head-controlled cursor state
    private var cursorCalibrated = false
    private var baseYaw = 0f
    private var basePitch = 0f
    private var cursorX = 0f
    private var cursorY = 0f

    // Finger flick (up -> next) state
    private val fingerHistory = ArrayDeque<Pair<Long, Float>>()
    private var fingerLastTrigger = 0L
    private var fingerArmed = true
    private var fingerMinD = 10f

    // Palm pose (open -> previous) state
    private var palmLastTrigger = 0L
    private var palmArmed = true

    private var lastDebug = 0L
    private val ui = Handler(Looper.getMainLooper())
    private var toast: Toast? = null

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
            val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            val opts = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
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
            handLandmarker = HandLandmarker.createFromOptions(this, opts)
        } catch (e: Exception) {
            Log.e(TAG, "Hand model load failed", e)
            say("Hand model load failed: " + e.message)
        }

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
            val hl = handLandmarker
            val fl = faceLandmarker
            if (hl == null && fl == null) return
            val bmp = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            imgW = rotated.width.toFloat()
            imgH = rotated.height.toFloat()
            val now = SystemClock.uptimeMillis()
            val mpImage = BitmapImageBuilder(rotated).build()
            hl?.detectAsync(mpImage, now)
            fl?.detectAsync(mpImage, now)
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
            say("analyze error: " + e.message)
        } finally {
            proxy.close()
        }
    }

    private fun onHandResult(result: HandLandmarkerResult) {
        val hand = result.landmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (hand == null) {
            if (handSeen && DEBUG) say("Hand lost")
            handSeen = false
            fingerHistory.clear()
            palmArmed = true
            return
        }
        if (!handSeen && DEBUG) say("Hand detected")
        handSeen = true

        // 0 = wrist, 9 = middle knuckle, 5 = index knuckle, 8 = index fingertip
        val palm = hypot(
            (hand[9].x() - hand[0].x()) * imgW,
            (hand[9].y() - hand[0].y()) * imgH
        )
        if (palm < 1f) return

        // ---- Finger flick up -> next reel ----
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
        } else if (now - fingerLastTrigger >= FINGER_COOLDOWN_MS && fingerHistory.size >= 2) {
            if (delta < -FINGER_THRESH) fireFinger(now)
        }

        // ---- Palm open -> previous reel / palm closed -> no action, just resets arming ----
        // A finger counts as "extended" when its tip sits farther from the wrist than its middle (PIP) joint.
        val fingerJoints = arrayOf(6 to 8, 10 to 12, 14 to 16, 18 to 20) // (PIP, TIP) for index/middle/ring/pinky
        var extendedCount = 0
        for ((pip, tip) in fingerJoints) {
            val dPip = hypot((hand[pip].x() - hand[0].x()) * imgW, (hand[pip].y() - hand[0].y()) * imgH)
            val dTip = hypot((hand[tip].x() - hand[0].x()) * imgW, (hand[tip].y() - hand[0].y()) * imgH)
            if (dTip > dPip * EXTENDED_RATIO) extendedCount++
        }
        if (DEBUG && now - lastDebug > 2500) {
            say("fingers extended=%d".format(extendedCount))
            lastDebug = now
        }
        if (extendedCount <= CLOSED_FINGER_COUNT) {
            palmArmed = true
        } else if (extendedCount >= OPEN_FINGER_COUNT && palmArmed && now - palmLastTrigger >= PALM_COOLDOWN_MS) {
            palmLastTrigger = now
            palmArmed = false
            if (DEBUG) say("PREVIOUS (palm open)")
            SwipeAccessibilityService.swipe(false)
        }
    }

    private fun fireFinger(now: Long) {
        fingerLastTrigger = now
        fingerArmed = false
        fingerMinD = 10f
        fingerHistory.clear()
        if (DEBUG) say("NEXT (finger up)")
        SwipeAccessibilityService.swipe(true)
    }

    private fun onFaceResult(result: FaceLandmarkerResult) {
        val face = result.faceLandmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (face == null) {
            eyesClosedSince = 0L
            closureHandled = false
            cursorCalibrated = false
            return
        }

        // 1 = nose tip, 10 = forehead, 152 = chin, 234 = right cheek, 454 = left cheek
        val nose = face[1]
        val forehead = face[10]
        val chin = face[152]
        val rightCheek = face[234]
        val leftCheek = face[454]
        val faceH = kotlin.math.abs(chin.y() - forehead.y())
        val faceW = kotlin.math.abs(leftCheek.x() - rightCheek.x())
        val havePose = faceH > 0.01f && faceW > 0.01f
        val pitchD = if (havePose) (nose.y() - (forehead.y() + chin.y()) / 2f) / faceH else 0f
        val yawD = if (havePose) (nose.x() - (leftCheek.x() + rightCheek.x()) / 2f) / faceW else 0f

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

        // ---- Long hold (>= 4s) -> show/hide cursor; short closures -> blink, 2 quick ones -> click ----
        if (eyesClosed) {
            if (eyesClosedSince == 0L) eyesClosedSince = now
            if (!closureHandled && now - eyesClosedSince >= EYES_CLOSED_HOLD_MS) {
                closureHandled = true
                ui.post { toggleCursor() }
            }
        } else {
            if (eyesClosedSince != 0L) {
                val closedFor = now - eyesClosedSince
                if (closedFor in BLINK_MIN_MS..BLINK_MAX_MS) {
                    blinkTimes.addLast(now)
                }
            }
            eyesClosedSince = 0L
            closureHandled = false
        }
        while (blinkTimes.isNotEmpty() && now - blinkTimes.first() > DOUBLE_BLINK_WINDOW_MS) blinkTimes.removeFirst()
        if (blinkTimes.size >= 2) {
            blinkTimes.clear()
            if (cursorView != null) ui.post { clickAtCursor() }
        }

        // ---- Head movement -> cursor movement (only while the cursor is showing) ----
        if (havePose && cursorView != null) {
            if (!cursorCalibrated) {
                baseYaw = yawD
                basePitch = pitchD
                cursorCalibrated = true
            } else {
                var offYaw = yawD - baseYaw
                if (MIRROR_CURSOR_X) offYaw = -offYaw
                val offPitch = pitchD - basePitch
                ui.post { moveCursor(offYaw, offPitch) }
            }
        }

        if (DEBUG && havePose && now - lastDebug > 2500) {
            say("pitch=%.2f yaw=%.2f eyes=%s".format(pitchD, yawD, if (eyesClosed) "closed" else "open"))
            lastDebug = now
        }
    }

    private fun toggleCursor() {
        if (cursorView != null) hideCursor() else showCursor()
    }

    private fun showCursor() {
        if (cursorView != null) return
        if (!Settings.canDrawOverlays(this)) {
            say("Enable \"Allow cursor overlay\" in the app first")
            return
        }
        try {
            val wm = getSystemService(WINDOW_SERVICE) as WindowManager
            val tv = TextView(this).apply {
                text = "\u27A4" // ➤ arrow
                textSize = 36f
                setTextColor(Color.WHITE)
                setShadowLayer(8f, 0f, 0f, Color.BLACK)
                rotation = -45f
            }
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            ).apply {
                gravity = Gravity.CENTER
                x = 0
                y = 0
            }
            wm.addView(tv, params)
            windowManager = wm
            cursorView = tv
            val dm = resources.displayMetrics
            cursorX = dm.widthPixels / 2f
            cursorY = dm.heightPixels / 2f
            cursorCalibrated = false // next head-pose reading becomes the new "center" reference
            if (DEBUG) say("Cursor shown (eyes closed 4s)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show cursor", e)
            say("Couldn't show cursor: " + e.message)
        }
    }

    private fun hideCursor() {
        val wm = windowManager
        val v = cursorView
        if (wm != null && v != null) {
            try {
                wm.removeView(v)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to remove cursor", e)
            }
        }
        cursorView = null
        cursorCalibrated = false
        if (DEBUG) say("Cursor hidden (eyes closed 4s)")
    }

    // Moves the cursor view based on how far the head has turned/tilted from its calibrated center.
    private fun moveCursor(offsetYaw: Float, offsetPitch: Float) {
        val tv = cursorView ?: return
        val wm = windowManager ?: return
        val dm = resources.displayMetrics
        var offX = offsetYaw * CURSOR_SENSITIVITY * dm.widthPixels
        var offY = offsetPitch * CURSOR_SENSITIVITY * dm.heightPixels
        offX = offX.coerceIn(-dm.widthPixels / 2f, dm.widthPixels / 2f)
        offY = offY.coerceIn(-dm.heightPixels / 2f, dm.heightPixels / 2f)
        cursorX = dm.widthPixels / 2f + offX
        cursorY = dm.heightPixels / 2f + offY
        try {
            val params = tv.layoutParams as WindowManager.LayoutParams
            params.x = offX.toInt()
            params.y = offY.toInt()
            wm.updateViewLayout(tv, params)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to move cursor", e)
        }
    }

    private fun clickAtCursor() {
        if (DEBUG) say("CLICK")
        SwipeAccessibilityService.tap(cursorX, cursorY)
    }

    override fun onDestroy() {
        hideCursor()
        handLandmarker?.close()
        faceLandmarker?.close()
        executor.shutdown()
        super.onDestroy()
    }
}
