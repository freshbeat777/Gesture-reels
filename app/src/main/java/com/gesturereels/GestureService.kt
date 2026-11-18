package com.gesturereels

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.drawable.GradientDrawable
import android.graphics.PixelFormat
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.View
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
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

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
        private const val OPEN_FINGER_COUNT = 4   // this many extended fingers (of 4) = palm open
        private const val CLOSED_FINGER_COUNT = 1 // this many or fewer extended = fist / closed, resets the arming

        // --- Eyes closed -> toggle cursor ---
        private const val EYES_CLOSED_HOLD_MS = 4000L // how long eyes must stay closed to toggle the cursor

        // --- Double blink -> click ---
        private const val BLINK_MIN_MS = 50L     // ignore flicker shorter than this
        private const val BLINK_MAX_MS = 500L    // a closure longer than this is a hold, not a blink
        private const val DOUBLE_BLINK_WINDOW_MS = 800L // two blinks within this window = a click
        private const val BLINK_FREEZE_SCORE = 0.25f    // eyelids this closed => cursor freezes (no jitter / wrong click spot)
        private const val POST_BLINK_FREEZE_MS = 150L   // cursor stays frozen briefly after eyes reopen

        // --- Head movement -> cursor movement ---
        // If moving your head left moves the cursor right (or vice versa), flip this.
        private const val MIRROR_CURSOR_X = true
        private const val CURSOR_Y_GAIN = 1.25f   // vertical head motion is smaller, so boost it a little
        private const val DEAD_ZONE = 0.008f      // tiny head tremor below this is ignored
        private const val CALIB_FRAMES = 8        // frames averaged to find your "center" when the cursor appears
        private const val ONE_EURO_BETA = 6f      // how quickly smoothing relaxes during fast head moves
        private const val CURSOR_SIZE_DP = 30

        // --- Performance ---
        private const val MAX_DIM = 480        // longest edge (px) fed to the ML models; smaller = faster
        private const val FACE_INTERVAL_MS = 30L // face model runs about every frame (blinks are very short)

        private const val DEBUG = true // show small status messages (set false to hide)
    }

    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private val executor = Executors.newSingleThreadExecutor()

    @Volatile private var imgW = 1f
    @Volatile private var imgH = 1f

    private var handSeen = false

    // Eyes-closed-hold / cursor show-hide state
    private var eyesClosedNow = false
    private var closeStart = 0L
    private var closureHandled = false
    @Volatile private var cursorView: View? = null
    private var windowManager: WindowManager? = null

    // Double-blink click state: (time, anchorX, anchorY) of each recent blink
    private val blinkEvents = ArrayDeque<Triple<Long, Float, Float>>()
    private var pendingAnchorX = 0f
    private var pendingAnchorY = 0f
    private var freezeUntil = 0L

    // Head-controlled cursor state
    @Volatile private var cursorCalibrated = false
    private var calibCount = 0
    private var calibSumYaw = 0f
    private var calibSumPitch = 0f
    private var baseYaw = 0f
    private var basePitch = 0f
    @Volatile private var cursorX = 0f
    @Volatile private var cursorY = 0f
    private var lastStableX = 0f
    private var lastStableY = 0f
    private val yawFilter = OneEuroFilter({ smoothCutoff() }, ONE_EURO_BETA)
    private val pitchFilter = OneEuroFilter({ smoothCutoff() }, ONE_EURO_BETA)

    // Live settings (changed from the app screen, applied instantly)
    private val prefs by lazy { getSharedPreferences("cfg", MODE_PRIVATE) }
    private fun cursorSpeed() = 1f + prefs.getInt("speed", 20) / 10f
    private fun smoothCutoff() = 3.0f - prefs.getInt("smooth", 6) * 0.27f
    private fun blinkCloseThreshold() = 0.75f - prefs.getInt("blink", 20) / 100f
    private fun pauseHandInCursor() = prefs.getBoolean("pause_hand", false)

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

    private var lastFaceRun = 0L

    private fun analyze(proxy: ImageProxy) {
        try {
            val hl = handLandmarker
            val fl = faceLandmarker
            if (hl == null && fl == null) return
            val bmp = proxy.toBitmap()
            val rotation = proxy.imageInfo.rotationDegrees
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            var rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)

            // Downscale — the models don't need full camera resolution, and this is the single
            // biggest cost per frame. Keeps aspect ratio, caps the long edge at MAX_DIM.
            val longEdge = maxOf(rotated.width, rotated.height)
            if (longEdge > MAX_DIM) {
                val scale = MAX_DIM.toFloat() / longEdge
                val w = (rotated.width * scale).toInt().coerceAtLeast(1)
                val h = (rotated.height * scale).toInt().coerceAtLeast(1)
                rotated = Bitmap.createScaledBitmap(rotated, w, h, true)
            }

            imgW = rotated.width.toFloat()
            imgH = rotated.height.toFloat()
            val now = SystemClock.uptimeMillis()
            val mpImage = BitmapImageBuilder(rotated).build()

            // Hand runs every frame (finger flicks are quick and need full responsiveness).
            hl?.detectAsync(mpImage, now)

            // Face only needs to run often enough to catch a blink or a 4-second hold —
            // throttling it to ~10fps cuts a large chunk of per-frame cost without hurting accuracy.
            if (fl != null && now - lastFaceRun >= FACE_INTERVAL_MS) {
                lastFaceRun = now
                fl.detectAsync(mpImage, now)
            }
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
        if (cursorView != null && pauseHandInCursor()) {
            fingerHistory.clear()
            palmArmed = true
            handSeen = false
            return
        }
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

        var fingerFiredThisFrame = false
        if (!fingerArmed) {
            if (d < fingerMinD) fingerMinD = d
            if (d - fingerMinD > FINGER_REARM_DELTA || now - fingerLastTrigger > FINGER_REARM_TIMEOUT_MS) {
                fingerArmed = true
                fingerHistory.clear()
            }
        } else if (now - fingerLastTrigger >= FINGER_COOLDOWN_MS && fingerHistory.size >= 2) {
            if (delta < -FINGER_THRESH) {
                fireFinger(now)
                fingerFiredThisFrame = true
            }
        }

        // ---- Palm open -> previous reel / palm closed -> no action, just resets arming ----
        // Skip entirely this frame (and briefly after) if the finger gesture just fired, so a
        // single flick that also spreads the other fingers can't ALSO register as an open palm.
        if (fingerFiredThisFrame || now - fingerLastTrigger < 400L) {
            if (DEBUG && now - lastDebug > 2500) {
                say("finger flick active, skipping palm check")
                lastDebug = now
            }
            return
        }

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

    private fun resetFaceState() {
        eyesClosedNow = false
        closureHandled = false
        blinkEvents.clear()
        cursorCalibrated = false
        calibCount = 0
        calibSumYaw = 0f
        calibSumPitch = 0f
        yawFilter.reset()
        pitchFilter.reset()
    }

    private fun deadZone(v: Float): Float {
        val a = abs(v)
        return if (a < DEAD_ZONE) 0f else (a - DEAD_ZONE) * (if (v < 0f) -1f else 1f)
    }

    private fun onFaceResult(result: FaceLandmarkerResult) {
        val face = result.faceLandmarks().firstOrNull()
        val now = SystemClock.uptimeMillis()
        if (face == null) {
            resetFaceState()
            return
        }
        val dm = resources.displayMetrics
        val screenW = dm.widthPixels.toFloat()
        val screenH = dm.heightPixels.toFloat()
        val w = imgW
        val h = imgH

        // ---------- Eye blink scores, with hysteresis (no flicker around the threshold) ----------
        var blinkL = 0f
        var blinkR = 0f
        val shapes = result.faceBlendshapes()
        if (shapes.isPresent && shapes.get().isNotEmpty()) {
            for (c in shapes.get()[0]) {
                when (c.categoryName()) {
                    "eyeBlinkLeft" -> blinkL = c.score()
                    "eyeBlinkRight" -> blinkR = c.score()
                }
            }
        }
        val lo = min(blinkL, blinkR)
        val hi = max(blinkL, blinkR)
        val avg = (blinkL + blinkR) / 2f
        val closeT = blinkCloseThreshold()
        val openT = (closeT - 0.2f).coerceAtLeast(0.15f)
        val wasClosed = eyesClosedNow
        if (!eyesClosedNow && lo > closeT) {
            eyesClosedNow = true      // BOTH eyes must close (a wink or squint is ignored)
        } else if (eyesClosedNow && hi < openT) {
            eyesClosedNow = false
        }

        // ---------- Head pose from landmarks, corrected for head roll (sideways tilt) ----------
        // 1 = nose tip, 10 = forehead, 152 = chin, 234/454 = cheeks, 33/263 = outer eye corners
        val nose = face[1]
        val forehead = face[10]
        val chin = face[152]
        val rightCheek = face[234]
        val leftCheek = face[454]
        val faceWpx = hypot((leftCheek.x() - rightCheek.x()) * w, (leftCheek.y() - rightCheek.y()) * h)
        val faceHpx = hypot((chin.x() - forehead.x()) * w, (chin.y() - forehead.y()) * h)
        val havePose = faceWpx > 1f && faceHpx > 1f
        var yawF = 0f
        var pitchF = 0f
        if (havePose) {
            val midX = (leftCheek.x() + rightCheek.x()) / 2f
            val midY = (forehead.y() + chin.y()) / 2f
            val dx = (nose.x() - midX) * w
            val dy = (nose.y() - midY) * h
            val roll = atan2((face[263].y() - face[33].y()) * h, (face[263].x() - face[33].x()) * w)
            val cr = cos(roll)
            val sr = sin(roll)
            val rx = dx * cr + dy * sr
            val ry = -dx * sr + dy * cr
            // One-Euro filter: smooth when the head is still, responsive when it moves fast
            yawF = yawFilter.filter(rx / faceWpx, now)
            pitchF = pitchFilter.filter(ry / faceHpx, now)
        }

        // ---------- Blink state machine ----------
        if (eyesClosedNow && !wasClosed) {
            closeStart = now
            closureHandled = false
            // The click spot = where the cursor was BEFORE the eyelids started to move
            pendingAnchorX = lastStableX
            pendingAnchorY = lastStableY
        }
        if (eyesClosedNow && !closureHandled && now - closeStart >= EYES_CLOSED_HOLD_MS) {
            closureHandled = true
            blinkEvents.clear()
            ui.post { toggleCursor() }
        }
        if (!eyesClosedNow && wasClosed) {
            freezeUntil = now + POST_BLINK_FREEZE_MS
            val dur = now - closeStart
            if (!closureHandled && dur in BLINK_MIN_MS..BLINK_MAX_MS) {
                blinkEvents.addLast(Triple(now, pendingAnchorX, pendingAnchorY))
            }
            closureHandled = false
        }
        while (blinkEvents.isNotEmpty() && now - blinkEvents.first().first > DOUBLE_BLINK_WINDOW_MS) {
            blinkEvents.removeFirst()
        }
        if (blinkEvents.size >= 2) {
            val first = blinkEvents.first()
            blinkEvents.clear()
            if (cursorView != null) ui.post { clickAt(first.second, first.third) }
        }

        // ---------- Head -> cursor (frozen while the eyes are closing / closed / just reopened) ----------
        val frozen = eyesClosedNow || avg > BLINK_FREEZE_SCORE || now < freezeUntil
        if (cursorView == null) {
            // cursor hidden: stay ready for the next time it is shown
            cursorCalibrated = false
            calibCount = 0
            calibSumYaw = 0f
            calibSumPitch = 0f
            lastStableX = screenW / 2f
            lastStableY = screenH / 2f
        } else if (havePose) {
            if (!cursorCalibrated) {
                if (!frozen) {
                    calibSumYaw += yawF
                    calibSumPitch += pitchF
                    calibCount++
                    if (calibCount >= CALIB_FRAMES) {
                        baseYaw = calibSumYaw / calibCount
                        basePitch = calibSumPitch / calibCount
                        cursorCalibrated = true
                    }
                }
            } else if (!frozen) {
                var offYaw = yawF - baseYaw
                if (MIRROR_CURSOR_X) offYaw = -offYaw
                val offPitch = pitchF - basePitch
                val speed = cursorSpeed()
                val offX = (deadZone(offYaw) * speed * screenW).coerceIn(-screenW / 2f, screenW / 2f)
                val offY = (deadZone(offPitch) * speed * CURSOR_Y_GAIN * screenH).coerceIn(-screenH / 2f, screenH / 2f)
                cursorX = screenW / 2f + offX
                cursorY = screenH / 2f + offY
                lastStableX = cursorX
                lastStableY = cursorY
                ui.post { applyCursor(offX, offY) }
            }
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
            val dm = resources.displayMetrics
            val size = (CURSOR_SIZE_DP * dm.density).toInt()
            val dot = View(this)
            dot.background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0x99FF3B30.toInt())
                setStroke((3 * dm.density).toInt(), Color.WHITE)
            }
            val params = WindowManager.LayoutParams(
                size,
                size,
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
            wm.addView(dot, params)
            windowManager = wm
            cursorX = dm.widthPixels / 2f
            cursorY = dm.heightPixels / 2f
            cursorCalibrated = false // the next few head readings become the new "center"
            cursorView = dot
            if (DEBUG) say("Cursor shown - hold your head still for a moment")
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
        if (DEBUG) say("Cursor hidden")
    }

    // Moves the cursor dot to (screen center + offset).
    private fun applyCursor(offX: Float, offY: Float) {
        val v = cursorView ?: return
        val wm = windowManager ?: return
        try {
            val params = v.layoutParams as WindowManager.LayoutParams
            params.x = offX.toInt()
            params.y = offY.toInt()
            wm.updateViewLayout(v, params)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to move cursor", e)
        }
    }

    private fun clickAt(x: Float, y: Float) {
        if (DEBUG) say("CLICK")
        SwipeAccessibilityService.tap(x, y)
    }

    override fun onDestroy() {
        hideCursor()
        handLandmarker?.close()
        faceLandmarker?.close()
        executor.shutdown()
        super.onDestroy()
    }
}

/**
 * One-Euro filter: a low-lag smoothing filter. Heavy smoothing while the signal is nearly still
 * (kills jitter), light smoothing while it moves fast (keeps the cursor responsive).
 */
class OneEuroFilter(private val minCutoff: () -> Float, private val beta: Float) {
    private var xPrev = 0f
    private var dxPrev = 0f
    private var tPrev = 0L
    private var ready = false

    fun reset() {
        ready = false
    }

    private fun alpha(cutoff: Float, dt: Float): Float {
        val tau = 1f / (2f * Math.PI.toFloat() * cutoff)
        return 1f / (1f + tau / dt)
    }

    fun filter(x: Float, tMs: Long): Float {
        if (!ready) {
            ready = true
            xPrev = x
            dxPrev = 0f
            tPrev = tMs
            return x
        }
        val dt = (tMs - tPrev).coerceAtLeast(1L) / 1000f
        tPrev = tMs
        val dx = (x - xPrev) / dt
        val aD = alpha(1.0f, dt)
        val dxHat = aD * dx + (1f - aD) * dxPrev
        dxPrev = dxHat
        val cutoff = minCutoff() + beta * abs(dxHat)
        val a = alpha(cutoff, dt)
        val xHat = a * x + (1f - a) * xPrev
        xPrev = xHat
        return xHat
    }
}
