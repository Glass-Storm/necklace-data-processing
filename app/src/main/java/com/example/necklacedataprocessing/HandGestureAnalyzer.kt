package com.example.necklacedataprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * On-device hand landmark detection and gesture classification.
 *
 * This is the "AI layer" the project was missing: the Pi relays JPEG frames and
 * this class interprets them. It runs entirely on the phone, which is what keeps
 * the project's 100% offline requirement intact — no cloud inference, no API keys.
 *
 * Pipeline:
 *   Bitmap (from the Pi's Base64 JPEG)
 *     -> MediaPipe HandLandmarker  -> 21 landmarks per hand
 *     -> finger-flexion angles     -> 5 booleans (extended / curled)
 *     -> pattern lookup            -> gesture name
 *
 * Why MediaPipe Tasks and not a custom CNN:
 *   The landmark model is already trained and shipped as a 7.8 MB asset, so the
 *   project's novel work is the gesture mapping and the integration, not
 *   re-deriving hand tracking from scratch. It also runs on CPU without a GPU
 *   delegate, so it works on mid-range handsets.
 *
 * Threading:
 *   [analyze] performs synchronous inference and must NOT be called on the UI
 *   thread. It is called from the network thread that already reads frames.
 *
 * Lifecycle:
 *   The native landmarker holds memory outside the JVM heap. Call [close] when
 *   the owner is destroyed or the object will leak native memory.
 */
class HandGestureAnalyzer(context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "HandGestureAnalyzer"

        /** Bundled model. Read from assets, so no download and no network. */
        private const val MODEL_ASSET = "hand_landmarker.task"

        /**
         * Minimum angle at a finger's middle joint for it to count as extended.
         *
         * A straight finger measures close to 180 degrees; a curled one drops
         * below 90. 150 sits well clear of both, leaving room for the landmark
         * jitter that MediaPipe shows on low-resolution frames.
         */
        private const val EXTENDED_ANGLE_DEGREES = 150.0

        /**
         * Frames a gesture must persist before it is reported as stable.
         *
         * At the Pi's 10 FPS a 5-frame window is half a second — long enough to
         * reject a single noisy frame, short enough that the readout still feels
         * responsive. Without this the label flickers between adjacent patterns.
         */
        private const val STABILITY_WINDOW = 5
        private const val STABILITY_REQUIRED = 3

        /** Reported while the recent frames disagree, i.e. the gesture is in flux. */
        private const val UNSTABLE = "—"

        /** Landmark indices, named so the geometry below stays readable. */
        private const val WRIST = 0
        private const val THUMB_CMC = 1
        private const val THUMB_MCP = 2
        private const val THUMB_IP = 3
        private const val THUMB_TIP = 4
        private const val INDEX_MCP = 5
        private const val INDEX_PIP = 6
        private const val INDEX_TIP = 8
        private const val MIDDLE_MCP = 9
        private const val MIDDLE_PIP = 10
        private const val MIDDLE_TIP = 12
        private const val RING_PIP = 14
        private const val RING_TIP = 16
        private const val PINKY_PIP = 18
        private const val PINKY_TIP = 20

        /**
         * Bone connections for the debug overlay, matching MediaPipe's own
         * HAND_CONNECTIONS constant.
         */
        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,          // thumb
            0 to 5, 5 to 6, 6 to 7, 7 to 8,          // index
            5 to 9, 9 to 10, 10 to 11, 11 to 12,     // middle
            9 to 13, 13 to 14, 14 to 15, 15 to 16,   // ring
            13 to 17, 17 to 18, 18 to 19, 19 to 20,  // pinky
            0 to 17,                                 // palm edge
        )
    }

    /** Which fingers the classifier judged to be extended. */
    data class FingerStates(
        val thumb: Boolean,
        val index: Boolean,
        val middle: Boolean,
        val ring: Boolean,
        val pinky: Boolean,
    ) {
        /** Number of extended fingers, 0-5. */
        val count: Int
            get() = listOf(thumb, index, middle, ring, pinky).count { it }

        /**
         * Compact 5-character code, thumb first, e.g. "01100" for a peace sign.
         *
         * The pattern is the classifier's real output — the gesture *name* is
         * only a label applied on top of it, which keeps renaming a word a
         * one-line change rather than a change to the geometry.
         */
        fun pattern(): String = buildString {
            append(if (thumb) '1' else '0')
            append(if (index) '1' else '0')
            append(if (middle) '1' else '0')
            append(if (ring) '1' else '0')
            append(if (pinky) '1' else '0')
        }
    }

    /**
     * The outcome of analysing one frame.
     *
     * @param gesture the classification for this frame alone
     * @param stableGesture the majority classification over the recent window
     * @param isStable true when [stableGesture] had enough support to be trusted
     * @param fingers which fingers were extended
     * @param handedness "Left"/"Right" as reported by the model
     * @param confidence the model's handedness score, 0-1
     */
    data class GestureResult(
        val gesture: String,
        val stableGesture: String,
        val isStable: Boolean,
        val fingers: FingerStates,
        val handedness: String,
        val confidence: Float,
    )

    private val landmarker: HandLandmarker = createLandmarker(context)

    /** Rolling window of recent raw classifications, for the stability filter. */
    private val recentGestures = ArrayDeque<String>()

    private var closed = false

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    private fun createLandmarker(context: Context): HandLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            // CPU is deliberate: the GPU delegate needs OpenGL ES 3.1 and fails
            // on some OEM drivers. CPU inference on a 320x240 frame is a few
            // milliseconds, so the compatibility is worth more than the speed.
            .setDelegate(Delegate.CPU)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            // IMAGE mode, not LIVE_STREAM. Frames arrive as independent JPEGs at
            // ~10 FPS, so there is no benefit to MediaPipe's temporal tracking,
            // and IMAGE mode is synchronous: it applies natural backpressure and
            // needs no monotonically increasing timestamps (which would break
            // whenever the Pi restarts and its clock resets).
            .setRunningMode(RunningMode.IMAGE)
            // One hand for now. Signing is dominated by single-hand shapes, and
            // limiting the search halves the inference cost.
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        Log.d(TAG, "Creating HandLandmarker from asset $MODEL_ASSET")
        return HandLandmarker.createFromOptions(context, options)
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /**
     * Detect and classify a hand in [bitmap].
     *
     * @return the result, or null when no hand was found in the frame.
     */
    fun analyze(bitmap: Bitmap): GestureResult? {
        if (closed) return null

        val result = try {
            landmarker.detect(BitmapImageBuilder(bitmap).build())
        } catch (e: Exception) {
            // A single malformed frame must not kill the streaming loop.
            Log.e(TAG, "Detection failed: ${e.message}")
            return null
        }

        val hands = result.landmarks()
        if (hands.isEmpty()) {
            // No hand in view. Clear the window so a stale gesture cannot be
            // reported as stable once the hand reappears somewhere else.
            recentGestures.clear()
            lastLandmarks = null
            return null
        }

        val landmarks = hands[0]
        if (landmarks.size < 21) {
            Log.w(TAG, "Expected 21 landmarks, got ${landmarks.size}")
            lastLandmarks = null
            return null
        }

        // Retained so the debug overlay can draw the skeleton for this frame.
        lastLandmarks = landmarks

        val fingers = readFingers(landmarks)
        val raw = classify(fingers)
        val stable = stabilise(raw)

        // Handedness is reported for the image as given. MediaPipe's convention
        // assumes a non-mirrored image, so a front-facing camera would invert
        // this label — the Pi's webcam is not mirrored, so it is used as-is.
        val category = result.handedness().firstOrNull()?.firstOrNull()
        val handedness = category?.categoryName() ?: "Unknown"
        val confidence = category?.score() ?: 0f

        return GestureResult(
            gesture = raw,
            stableGesture = stable,
            isStable = stable == raw && stable != UNSTABLE,
            fingers = fingers,
            handedness = handedness,
            confidence = confidence,
        )
    }

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------

    /**
     * Decide which fingers are extended, using the flexion angle at each
     * finger's middle joint.
     *
     * Joint angles are used rather than raw coordinates because they are
     * invariant to both hand rotation and distance from the camera — a fist
     * reads the same whether the hand is upright, sideways, near or far. A
     * simpler test such as "is the fingertip above the knuckle" breaks as soon
     * as the user tilts their hand.
     *
     * The thumb is measured at its MCP joint (1-2-3), NOT its IP joint (2-3-4).
     *
     * This was verified against MediaPipe's reference landmark files for known
     * gestures (tools/verify_geometry.py). At the IP joint a thumbs-up reads
     * 158.6 degrees and a victory sign 158.2 — a 0.4 degree gap, so no threshold
     * can separate them and a peace sign misclassifies as "11100". At the MCP
     * joint the same two gestures read 165.3 and 113.7, a 37 degree gap.
     * All four reference gestures classify correctly with this joint.
     */
    private fun readFingers(lm: List<NormalizedLandmark>): FingerStates {
        return FingerStates(
            thumb = angleAt(lm[THUMB_CMC], lm[THUMB_MCP], lm[THUMB_IP]) > EXTENDED_ANGLE_DEGREES,
            index = angleAt(lm[INDEX_MCP], lm[INDEX_PIP], lm[INDEX_TIP]) > EXTENDED_ANGLE_DEGREES,
            middle = angleAt(lm[MIDDLE_MCP], lm[MIDDLE_PIP], lm[MIDDLE_TIP]) > EXTENDED_ANGLE_DEGREES,
            ring = angleAt(lm[13], lm[RING_PIP], lm[RING_TIP]) > EXTENDED_ANGLE_DEGREES,
            pinky = angleAt(lm[17], lm[PINKY_PIP], lm[PINKY_TIP]) > EXTENDED_ANGLE_DEGREES,
        )
    }

    /**
     * Interior angle at [b], in degrees, formed by the segments b->a and b->c.
     *
     * The angle is computed in 3D, including the landmarks' Z coordinate.
     *
     * A 2D angle (X/Y only) fails when a finger curls toward or away from the
     * camera: the curl is then mostly in Z, and the 2D projection of the finger
     * chain can look nearly straight. A side-view fist was observed reading as
     * "all fingers extended" on the emulator for exactly this reason. Including
     * Z restores the curl. Verified against MediaPipe's reference landmarks:
     * all four reference gestures still classify correctly in 3D
     * (tools/verify_3d.py).
     *
     * Returns 180 for a perfectly straight joint and approaches 0 as it folds
     * back on itself.
     */
    private fun angleAt(
        a: NormalizedLandmark,
        b: NormalizedLandmark,
        c: NormalizedLandmark,
    ): Double {
        // Widened to Double: NormalizedLandmark exposes Float coordinates, but
        // acos() needs Double and the extra precision costs nothing here.
        val abX = (a.x() - b.x()).toDouble()
        val abY = (a.y() - b.y()).toDouble()
        val abZ = (a.z() - b.z()).toDouble()
        val cbX = (c.x() - b.x()).toDouble()
        val cbY = (c.y() - b.y()).toDouble()
        val cbZ = (c.z() - b.z()).toDouble()

        val magnitudeAB = sqrt(abX * abX + abY * abY + abZ * abZ)
        val magnitudeCB = sqrt(cbX * cbX + cbY * cbY + cbZ * cbZ)

        // Coincident landmarks carry no angle information.
        if (magnitudeAB == 0.0 || magnitudeCB == 0.0) return 0.0

        val cosine = ((abX * cbX + abY * cbY + abZ * cbZ) / (magnitudeAB * magnitudeCB))
            .coerceIn(-1.0, 1.0)

        return Math.toDegrees(acos(cosine))
    }

    // -----------------------------------------------------------------------
    // Classification
    // -----------------------------------------------------------------------

    /**
     * Map a finger pattern to a name.
     *
     * Patterns are written thumb-first. The names are descriptive labels for
     * verification, not final vocabulary — mapping them to words and phrases is
     * the next milestone, and it only requires editing this table.
     */
    private fun classify(f: FingerStates): String = when (f.pattern()) {
        "00000" -> "Fist"
        "11111" -> "Open palm"
        "10000" -> "Thumbs up"
        "01000" -> "Point"
        "11000" -> "L shape"
        "01100" -> "Peace"
        "01110" -> "Three"
        "01111" -> "Four"
        "01001" -> "Horns"
        "11001" -> "I love you"
        "00001" -> "Pinky"
        "11100" -> "Three (thumb out)"
        "11110" -> "Four (thumb out)"
        "10001" -> "Call me"
        else -> "Unknown ${f.pattern()}"
    }

    /**
     * Suppress single-frame flicker by requiring a gesture to repeat.
     *
     * Landmark jitter on a 320x240 frame occasionally flips one finger's state
     * for a single frame. Reporting the majority over the last few frames keeps
     * the displayed label steady without adding perceptible lag.
     */
    private fun stabilise(raw: String): String {
        recentGestures.addLast(raw)
        while (recentGestures.size > STABILITY_WINDOW) {
            recentGestures.removeFirst()
        }

        val votes = recentGestures.groupingBy { it }.eachCount()
        val winner = votes.maxByOrNull { it.value } ?: return raw

        return if (winner.value >= STABILITY_REQUIRED) winner.key else UNSTABLE
    }

    // -----------------------------------------------------------------------
    // Debug overlay
    // -----------------------------------------------------------------------

    /**
     * Draw the detected skeleton onto a copy of [source].
     *
     * Purely for verification: it makes a correct detection visually obvious on
     * a screenshot, which is the only practical way to confirm the pipeline
     * works when the Pi and the phone are not both on the bench.
     *
     * @return a new bitmap, or null when no hand was detected.
     */
    fun drawOverlay(source: Bitmap, landmarks: List<NormalizedLandmark>?): Bitmap? {
        if (landmarks == null || landmarks.size < 21) return null

        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val width = output.width.toFloat()
        val height = output.height.toFloat()

        val bonePaint = Paint().apply {
            color = Color.CYAN
            strokeWidth = 3f
            isAntiAlias = true
        }
        val jointPaint = Paint().apply {
            color = Color.YELLOW
            strokeWidth = 6f
            isAntiAlias = true
        }

        for ((from, to) in HAND_CONNECTIONS) {
            canvas.drawLine(
                landmarks[from].x() * width, landmarks[from].y() * height,
                landmarks[to].x() * width, landmarks[to].y() * height,
                bonePaint,
            )
        }
        for (landmark in landmarks) {
            canvas.drawPoint(landmark.x() * width, landmark.y() * height, jointPaint)
        }

        return output
    }

    /**
     * Expose the raw landmarks of the most recent [analyze] call for the overlay.
     *
     * Kept separate from [GestureResult] so the classifier's output stays a
     * clean, UI-agnostic data class.
     */
    var lastLandmarks: List<NormalizedLandmark>? = null
        private set

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /** Release the native model. Safe to call more than once. */
    override fun close() {
        if (closed) return
        closed = true
        try {
            landmarker.close()
            Log.d(TAG, "HandLandmarker released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing landmarker: ${e.message}")
        }
    }
}
