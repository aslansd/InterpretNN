package com.example

import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

// Structure to persist layer activations for analysis
data class ModelActivations(
    val cnnConv1: FloatArray,   // flattened Conv1 activations (dim=16)
    val cnnDense: FloatArray,   // CNN Dense layer activations (dim=8)
    val dqnDense1: FloatArray,  // DQN Dense 1 activations (dim=8)
    val dqnDense2: FloatArray,  // DQN Dense 2 (Q-values) (dim=3)
    val dtEmbed: FloatArray,    // DT State Embedding of current step (dim=4)
    val dtAttention: FloatArray // DT Attention weights (flat $5 \times 5$ for Head 1)
)

object NeuralModels {

    // --- CNN Model Variables ---
    // Conv1: 4 filters of 3x3
    private val conv1Kernels = arrayOf(
        // Filter 0: Paddle tracker (vertical-ish edges)
        arrayOf(
            floatArrayOf(-1f, 2f, -1f),
            floatArrayOf(-1f, 2f, -1f),
            floatArrayOf(-1f, 2f, -1f)
        ),
        // Filter 1: Ball tracker (point/center focus)
        arrayOf(
            floatArrayOf(-1f, -1f, -1f),
            floatArrayOf(-1f,  8f, -1f),
            floatArrayOf(-1f, -1f, -1f)
        ),
        // Filter 2: Brick tracker (horizontal-ish structures)
        arrayOf(
            floatArrayOf( 2f,  2f,  2f),
            floatArrayOf(-1f, -1f, -1f),
            floatArrayOf(-1f, -1f, -1f)
        ),
        // Filter 3: General identity / blur
        arrayOf(
            floatArrayOf( 0.2f, 0.2f, 0.2f),
            floatArrayOf( 0.2f, 0.2f, 0.2f),
            floatArrayOf( 0.2f, 0.2f, 0.2f)
        )
    )

    // CNN Dense Weights (Flatten size is 8x8x4 = 256 mapped to 8, then 8 mapped to 3)
    // We'll simulate a 2-stage dense: 256 -> 8 (Dense 1), then 8 -> 3 (Logits)
    private val cnnW1 = Array(8) { FloatArray(256) { 0f } }
    private val cnnB1 = FloatArray(8) { 0f }
    private val cnnW2 = Array(3) { FloatArray(8) { 0f } }
    private val cnnB2 = FloatArray(3) { 0f }

    // --- DQN (RAM) Model Variables ---
    // Input: 6 values. Hidden: 8, Output: 3
    private val dqnW1 = Array(8) { FloatArray(6) { 0f } }
    private val dqnB1 = FloatArray(8) { 0f }
    private val dqnW2 = Array(3) { FloatArray(8) { 0f } }
    private val dqnB2 = FloatArray(3) { 0f }

    // --- Decision Transformer (DT) Variables ---
    // Sequence size: 5 steps. Dimension of State: 3, Action: 3, RTG: 1
    // We project each component to embed size of 4.
    private val dtStateW = Array(4) { FloatArray(3) { 0f } }
    private val dtActionW = Array(4) { FloatArray(3) { 0f } }
    private val dtRtgW = Array(4) { FloatArray(1) { 0f } }
    
    // Self attention weights for Head 1 and Head 2. Query/Key/Value projections: 4 -> 2
    private val dtWq1 = Array(2) { FloatArray(4) { 0f } }
    private val dtWk1 = Array(2) { FloatArray(4) { 0f } }
    private val dtWv1 = Array(2) { FloatArray(4) { 0f } }
    
    private val dtWq2 = Array(2) { FloatArray(4) { 0f } }
    private val dtWk2 = Array(2) { FloatArray(4) { 0f } }
    private val dtWv2 = Array(2) { FloatArray(4) { 0f } }

    private val dtDenseW = Array(3) { FloatArray(4) { 0f } }

    init {
        initializeStructuredWeights()
    }

    private fun initializeStructuredWeights() {
        // Initialize weights to emulate cohesive policies (rather than random chaos)
        
        // 1) CNN Policy: Emulate ball tracking and brick visual alignment
        // W1 intercepts local positions.
        for (i in 0 until 8) {
            for (j in 0 until 256) {
                val f = j % 4
                val cellIndex = j / 4
                val r = cellIndex / 8
                val c = cellIndex % 8
                
                // Let filter 1 (ball tracker) influence the decision to move paddle
                if (f == 1) {
                    // Left tracking mapping
                    if (c < 4) {
                        cnnW1[0][j] = 0.12f  // neuron 0 responds to ball on left
                        cnnW1[1][j] = -0.08f
                    } else {
                        cnnW1[0][j] = -0.08f
                        cnnW1[1][j] = 0.12f  // neuron 1 responds to ball on right
                    }
                }
            }
        }
        
        // Direct dense outputs
        cnnW2[0][0] = 2.5f  // Logit Left: highly active when ball on left is detected
        cnnW2[1][1] = 2.5f  // Logit Right: highly active when ball on right is detected
        cnnW2[2][2] = 0.5f  // Logit Stay

        // 2) DQN Policy: Emulate explicit ball alignment
        // Inputs: [paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_count]
        // Neuron 0: Ball is to the RIGHT of paddle (need to move right)
        dqnW1[0][0] = -3.5f // paddle_x coeff
        dqnW1[0][1] = 3.5f  // ball_x coeff
        dqnW1[0][2] = 0.1f
        dqnB1[0] = -0.2f

        // Neuron 1: Ball is to the LEFT of paddle (need to move left)
        dqnW1[1][0] = 3.5f
        dqnW1[1][1] = -3.5f
        dqnW1[1][2] = 0.1f
        dqnB1[1] = -0.2f

        // Neuron 2: Keep calm when ball and paddle are aligned
        dqnW1[2][0] = -0.5f
        dqnW1[2][1] = 0.5f
        dqnW1[2][2] = -0.5f

        // Map hidden outputs to actions: 0=Left, 1=Right, 2=Stay
        dqnW2[0][1] = 3.2f  // Action Left active when Neuron 1 is active (ball is left of paddle)
        dqnW2[1][0] = 3.2f  // Action Right active when Neuron 0 is active (ball is right of paddle)
        dqnW2[2][2] = 1.0f  // Action Stay
        
        // 3) DT Policy: Emulate state embedding project and attention weights
        // Projections to size 4
        // State: [paddle, ball_x, ball_y]
        dtStateW[0][0] = -2.0f; dtStateW[0][1] = 2.0f   // dimension 0: ball_x - paddle
        dtStateW[1][0] = 2.0f;  dtStateW[1][1] = -2.0f  // dimension 1: paddle - ball_x
        dtStateW[2][2] = 1.5f                           // dimension 2: vertical location
        dtStateW[3][1] = 1.0f                           // dimension 3: horizontal ball

        // Attention matrices (Head 1 attends heavily to current state, Head 2 is historic)
        for (i in 0..1) {
            dtWq1[i][i] = 1.0f; dtWk1[i][i] = 1.0f; dtWv1[i][i] = 1.0f
            dtWq2[i][i] = 0.5f; dtWk2[i][3-i] = 1.2f; dtWv2[i][i] = 1.0f
        }
        
        // Output weights
        dtDenseW[0][1] = 4.0f  // Move left
        dtDenseW[1][0] = 4.0f  // Move right
        dtDenseW[2][2] = 1.0f  // Stay
    }

    // ==========================================
    // 1) CNN FORWARD + BACKPROPAGATION (Saliency Map)
    // ==========================================
    fun runCnnWithSaliency(input: Array<FloatArray>): Pair<Int, Array<FloatArray>> {
        // Forward conv1 (16x16 -> 16x16x4 with padding=1, stride=1)
        val convOut = Array(16) { Array(16) { FloatArray(4) { 0f } } }
        for (f in 0 until 4) {
            val k = conv1Kernels[f]
            for (r in 0 until 16) {
                for (c in 0 until 16) {
                    var sum = 0f
                    for (ki in 0 until 3) {
                        for (kj in 0 until 3) {
                            val ir = r + ki - 1
                            val ic = c + kj - 1
                            if (ir in 0..15 && ic in 0..15) {
                                sum += input[ir][ic] * k[ki][kj]
                            }
                        }
                    }
                    convOut[r][c][f] = if (sum > 0f) sum else 0f // ReLU
                }
            }
        }

        // Max Pool 2x2 (16x16x4 -> 8x8x4)
        val poolOut = Array(8) { Array(8) { FloatArray(4) { 0f } } }
        val poolMaxIndices = Array(8) { Array(8) { Array(4) { Pair(0, 0) } } }
        for (f in 0 until 4) {
            for (pr in 0 until 8) {
                for (pc in 0 until 8) {
                    var maxVal = -1e9f
                    var maxR = pr * 2
                    var maxC = pc * 2
                    for (dr in 0 until 2) {
                        for (dc in 0 until 2) {
                            val r = pr * 2 + dr
                            val c = pc * 2 + dc
                            if (convOut[r][c][f] > maxVal) {
                                maxVal = convOut[r][c][f]
                                maxR = r
                                maxC = c
                            }
                        }
                    }
                    poolOut[pr][pc][f] = maxVal
                    poolMaxIndices[pr][pc][f] = Pair(maxR, maxC)
                }
            }
        }

        // Flatten (8x8x4 -> 256)
        val flattened = FloatArray(256)
        var flatIdx = 0
        for (pr in 0 until 8) {
            for (pc in 0 until 8) {
                for (f in 0 until 4) {
                    flattened[flatIdx++] = poolOut[pr][pc][f]
                }
            }
        }

        // Dense Layer 1 (256 -> 8)
        val dense1Act = FloatArray(8)
        for (i in 0 until 8) {
            var sum = cnnB1[i]
            for (j in 0 until 256) {
                sum += flattened[j] * cnnW1[i][j]
            }
            dense1Act[i] = if (sum > 0f) sum else 0f // ReLU
        }

        // Dense Layer 2 Logits (8 -> 3)
        val logits = FloatArray(3)
        for (i in 0 until 3) {
            var sum = cnnB2[i]
            for (j in 0 until 8) {
                sum += dense1Act[j] * cnnW2[i][j]
            }
            logits[i] = sum
        }

        // Choose Action
        val action = argMax(logits)

        // --- BACKPROPAGATION FOR SALIENCY MAP ---
        // Saliency is visualised as d(logit[chosen_action]) / d(input_pixels)
        // dLogit/dDense1
        val dDense1Act = FloatArray(8)
        for (i in 0 until 8) {
            dDense1Act[i] = cnnW2[action][i]
        }
        
        // dDense1/dFlattened
        val dFlattened = FloatArray(256) { 0f }
        for (i in 0 until 8) {
            if (dense1Act[i] > 0f) { // pass ReLU
                for (j in 0 until 256) {
                    dFlattened[j] += dDense1Act[i] * cnnW1[i][j]
                }
            }
        }

        // dFlattened/dConvOut
        val dConvOut = Array(16) { Array(16) { FloatArray(4) { 0f } } }
        flatIdx = 0
        for (pr in 0 until 8) {
            for (pc in 0 until 8) {
                for (f in 0 until 4) {
                    val dFlat = dFlattened[flatIdx++]
                    val (maxR, maxC) = poolMaxIndices[pr][pc][f]
                    dConvOut[maxR][maxC][f] = dFlat
                }
            }
        }

        // dConvOut/dInput
        val dInput = Array(16) { FloatArray(16) { 0f } }
        for (f in 0 until 4) {
            val k = conv1Kernels[f]
            for (r in 0 until 16) {
                for (c in 0 until 16) {
                    val dConv = dConvOut[r][c][f]
                    if (dConv != 0f && convOut[r][c][f] > 0f) { // pass ReLU
                        for (ki in 0 until 3) {
                            for (kj in 0 until 3) {
                                val ir = r + ki - 1
                                val ic = c + kj - 1
                                if (ir in 0..15 && ic in 0..15) {
                                    dInput[ir][ic] += dConv * k[ki][kj]
                                }
                            }
                        }
                    }
                }
            }
        }

        // Postprocess saliency (absolute values, normalized)
        val saliency = Array(16) { r ->
            FloatArray(16) { c ->
                kotlin.math.abs(dInput[r][c])
            }
        }
        normalizeMatrix(saliency)

        return Pair(action, saliency)
    }

    // ==========================================
    // 2) DQN RAM MODEL FORWARD
    // ==========================================
    fun runDqnRam(input6: FloatArray): Pair<Int, FloatArray> {
        // Input: 6 continuous variables
        // Dense 1 (6 -> 8)
        val act1 = FloatArray(8)
        for (i in 0 until 8) {
            var sum = dqnB1[i]
            for (j in 0 until 6) {
                sum += input6[j] * dqnW1[i][j]
            }
            act1[i] = if (sum > 0f) sum else 0f // ReLU
        }

        // Dense 2 (8 -> 3)
        val qValues = FloatArray(3)
        for (i in 0 until 3) {
            var sum = dqnB2[i]
            for (j in 0 until 8) {
                sum += act1[j] * dqnW2[i][j]
            }
            qValues[i] = sum
        }
        
        return Pair(argMax(qValues), act1)
    }

    // ==========================================
    // 3) DECISION TRANSFORMER FORWARD (Sequence + Attention)
    // ==========================================
    fun runDecisionTransformer(sequence: List<SequenceStep>): Pair<Int, Array<FloatArray>> {
        val k = sequence.size
        // Embeddings: aggregate components of each step (dim=4)
        val embeddings = Array(k) { FloatArray(4) { 0f } }
        for (t in 0 until k) {
            val step = sequence[t]
            val stateEmbed = projectVector(step.state, dtStateW)
            val actionOneHot = FloatArray(3) { if (it == step.action) 1.0f else 0.0f }
            val actionEmbed = projectVector(actionOneHot, dtActionW)
            val rtgEmbed = projectVector(floatArrayOf(step.rtg), dtRtgW)
            
            // Compose token representation
            for (d in 0 until 4) {
                embeddings[t][d] = stateEmbed[d] + actionEmbed[d] + rtgEmbed[d] + (t * 0.05f) // Positional bias
            }
        }

        // Multi-head Attention (Head 1: causal key-query matches)
        val q1 = Array(k) { projectDim4To2(embeddings[it], dtWq1) }
        val k1 = Array(k) { projectDim4To2(embeddings[it], dtWk1) }
        val v1 = Array(k) { projectDim4To2(embeddings[it], dtWv1) }

        // Compute attention scores [k x k]
        val attentionWeights1 = Array(k) { FloatArray(k) { 0f } }
        for (i in 0 until k) {
            val rowExp = FloatArray(k)
            var sumExp = 0f
            for (j in 0 until k) {
                if (j <= i) { // Causal Mask
                    var dot = q1[i][0] * k1[j][0] + q1[i][1] * k1[j][1]
                    dot /= sqrt(2.0f) // Scaled dot-product
                    rowExp[j] = exp(dot.coerceIn(-10f, 10f))
                    sumExp += rowExp[j]
                } else {
                    rowExp[j] = 0f
                }
            }
            for (j in 0 until k) {
                attentionWeights1[i][j] = if (sumExp > 0f) rowExp[j]/sumExp else 0f
            }
        }

        // Apply attention output weights
        val attentionOutput = FloatArray(4) { 0f }
        // Aggregate last token representations
        val lastIdx = k - 1
        for (j in 0 until k) {
            val w = attentionWeights1[lastIdx][j]
            val valVec = v1[j] // size 2
            attentionOutput[0] += w * valVec[0]
            attentionOutput[1] += w * valVec[1]
            attentionOutput[2] += w * 0.5f
            attentionOutput[3] += w * 0.1f
        }

        // Logits via Dense Projection
        val logits = FloatArray(3)
        for (i in 0 until 3) {
            var sum = 0f
            for (j in 0 until 4) {
                sum += attentionOutput[j] * dtDenseW[i][j]
            }
            logits[i] = sum
        }

        return Pair(argMax(logits), attentionWeights1)
    }

    // ==========================================
    // 4) CKA REPRESENTATION COMPARATOR
    // ==========================================
    // Takes a batch (N x D) of hidden state records, aligns matrices, centered HSIC
    fun computeCka(activationsX: Array<FloatArray>, activationsY: Array<FloatArray>): Float {
        val n = activationsX.size
        if (n <= 1) return 1.0f
        
        // Compute GRAM Matrices: K = X X^T, L = Y Y^T
        val kMat = Array(n) { FloatArray(n) { 0f } }
        val lMat = Array(n) { FloatArray(n) { 0f } }
        
        for (i in 0 until n) {
            for (j in 0 until n) {
                kMat[i][j] = dotProduct(activationsX[i], activationsX[j])
                lMat[i][j] = dotProduct(activationsY[i], activationsY[j])
            }
        }
        
        // Centering operator: H = I - 1/n
        // Centered matrix computed as: Kc = H K H => Kc[i][j] = K[i][j] - meanRow[i] - meanCol[j] + meanGrand
        val kCentered = centerGramMatrix(kMat)
        val lCentered = centerGramMatrix(lMat)
        
        // HSIC = trace(Kc Lc) = sum_ij Kc[i][j] * Lc[i][j]
        val hsicXY = frobeniusInnerProduct(kCentered, lCentered)
        val hsicXX = frobeniusInnerProduct(kCentered, kCentered)
        val hsicYY = frobeniusInnerProduct(lCentered, lCentered)
        
        if (hsicXX <= 0f || hsicYY <= 0f) return 0f
        return (hsicXY / sqrt(hsicXX * hsicYY)).coerceIn(0f, 1.0f)
    }

    private fun centerGramMatrix(k: Array<FloatArray>): Array<FloatArray> {
        val n = k.size
        val rowMeans = FloatArray(n)
        var grandMean = 0f
        
        for (i in 0 until n) {
            var rSum = 0f
            for (j in 0 until n) {
                rSum += k[i][j]
            }
            rowMeans[i] = rSum / n
            grandMean += rSum
        }
        grandMean /= (n * n)
        
        val centered = Array(n) { FloatArray(n) }
        for (i in 0 until n) {
            for (j in 0 until n) {
                centered[i][j] = k[i][j] - rowMeans[i] - rowMeans[j] + grandMean
            }
        }
        return centered
    }
    
    // --- Helper math routines ---
    private fun argMax(arr: FloatArray): Int {
        var maxIdx = 0
        var maxVal = -1e9f
        for (i in arr.indices) {
            if (arr[i] > maxVal) {
                maxVal = arr[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    private fun projectVector(vec: FloatArray, weights: Array<FloatArray>): FloatArray {
        val outSize = weights.size
        val outVec = FloatArray(outSize)
        for (i in 0 until outSize) {
            var sum = 0f
            for (j in vec.indices) {
                if (j < weights[i].size) {
                    sum += vec[j] * weights[i][j]
                }
            }
            outVec[i] = if (sum > 0f) sum else sum * 0.1f // LeakyReLU default
        }
        return outVec
    }

    private fun projectDim4To2(vec: FloatArray, weights: Array<FloatArray>): FloatArray {
        val out = FloatArray(2)
        for (i in 0..1) {
            var sum = 0f
            for (j in 0..3) {
                sum += vec[j] * weights[i][j]
            }
            out[i] = sum
        }
        return out
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        val size = kotlin.math.min(a.size, b.size)
        for (i in 0 until size) {
            sum += a[i] * b[i]
        }
        return sum
    }

    private fun frobeniusInnerProduct(a: Array<FloatArray>, b: Array<FloatArray>): Float {
        var sum = 0f
        val n = a.size
        for (i in 0 until n) {
            for (j in 0 until n) {
                sum += a[i][j] * b[i][j]
            }
        }
        return sum
    }

    private fun normalizeMatrix(matrix: Array<FloatArray>) {
        var maxVal = 1e-9f
        for (r in matrix.indices) {
            for (c in matrix[r].indices) {
                if (matrix[r][c] > maxVal) maxVal = matrix[r][c]
            }
        }
        for (r in matrix.indices) {
            for (c in matrix[r].indices) {
                matrix[r][c] /= maxVal
            }
        }
    }

    // --- High-level probe prediction values ---
    // Decodes continuous variables from CNN Conv, CNN Dense state, or DT Embedding
    fun runProbeDecoder(activations: FloatArray, attribute: String): Float {
        // Attribute is "paddleX", "ballX", "ballY"
        // Let's create a real deterministic linear probe model
        var sum = 0f
        when (attribute) {
            "paddleX" -> {
                // Decode from activations with simple linear weights
                for (i in activations.indices) {
                    val weight = if (i % 2 == 0) 0.15f else -0.1f
                    sum += activations[i] * weight
                }
                return (sum + 0.4f).coerceIn(0f, 1f)
            }
            "ballX" -> {
                for (i in activations.indices) {
                    val weight = if (i % 3 == 0) 0.2f else -0.05f
                    sum += activations[i] * weight
                }
                return (sum + 0.5f).coerceIn(0f, 1f)
            }
            "ballY" -> {
                for (i in activations.indices) {
                    val weight = if (i % 4 == 0) 0.18f else -0.08f
                    sum += activations[i] * weight
                }
                return (sum + 0.35f).coerceIn(0f, 1f)
            }
            else -> return 0.5f
        }
    }
}
