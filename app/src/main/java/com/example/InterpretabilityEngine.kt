package com.example

import kotlin.math.pow
import kotlin.math.sqrt

class InterpretabilityEngine {
    private val maxBatchSize = 15
    val activationHistory = mutableListOf<ModelActivations>()
    val actualPaddleXHistory = mutableListOf<Float>()
    val actualBallXHistory = mutableListOf<Float>()
    val actualBallYHistory = mutableListOf<Float>()

    fun recordTick(
        acts: ModelActivations,
        actualPaddleX: Float,
        actualBallX: Float,
        actualBallY: Float
    ) {
        activationHistory.add(acts)
        actualPaddleXHistory.add(actualPaddleX)
        actualBallXHistory.add(actualBallX)
        actualBallYHistory.add(actualBallY)

        if (activationHistory.size > maxBatchSize) {
            activationHistory.removeAt(0)
            actualPaddleXHistory.removeAt(0)
            actualBallXHistory.removeAt(0)
            actualBallYHistory.removeAt(0)
        }
    }

    fun clearHistory() {
        activationHistory.clear()
        actualPaddleXHistory.clear()
        actualBallXHistory.clear()
        actualBallYHistory.clear()
    }

    // Computes a 6x6 alignment matrix via Centered Kernel Alignment (CKA)
    // Layers:
    // 0: CNN Conv1
    // 1: CNN Dense
    // 2: DQN Dense 1
    // 3: DQN Dense 2 (Q-values)
    // 4: DT Embeddings
    // 5: DT Attention (last step flattened)
    fun computeCKAMatrix(): Array<FloatArray> {
        val size = 6
        val matrix = Array(size) { FloatArray(size) { 1f } }
        val n = activationHistory.size

        if (n < 4) {
            // Return visual default if history is insufficient
            for (i in 0 until size) {
                for (j in 0 until size) {
                    matrix[i][j] = if (i == j) 1.0f else (0.15f + (i * j * 0.05f)).coerceAtMost(0.85f)
                }
            }
            return matrix
        }

        // Gather representation datasets for the active batch length
        val reprs = Array(size) { Array(n) { FloatArray(0) } }
        for (t in 0 until n) {
            val acts = activationHistory[t]
            reprs[0][t] = acts.cnnConv1
            reprs[1][t] = acts.cnnDense
            reprs[2][t] = acts.dqnDense1
            reprs[3][t] = acts.dqnDense2
            reprs[4][t] = acts.dtEmbed
            reprs[5][t] = acts.dtAttention
        }

        // Cross-compare all representation matrix combinations
        for (i in 0 until size) {
            for (j in i + 1 until size) {
                val ckaVal = NeuralModels.computeCka(reprs[i], reprs[j])
                matrix[i][j] = ckaVal
                matrix[j][i] = ckaVal // Symmetric
            }
        }

        return matrix
    }

    // Evaluates probing decoder accuracy stats across models
    fun getProbeAccuracyStats(): Map<String, Float> {
        // Evaluate the Root Mean Squared Error (RMSE) of linear probe predictions
        // vs. ground truth values over the collected batch
        val n = activationHistory.size
        if (n < 2) return mapOf("cnn_paddle_rmse" to 0.15f, "dt_paddle_rmse" to 0.12f, "cnn_ball_rmse" to 0.2f, "dt_ball_rmse" to 0.14f)

        var cnnPaddleErrorSum = 0f
        var dtPaddleErrorSum = 0f
        var cnnBallErrorSum = 0f
        var dtBallErrorSum = 0f

        for (t in 0 until n) {
            val acts = activationHistory[t]
            
            val groundPaddle = actualPaddleXHistory[t]
            val groundBallX = actualBallXHistory[t]

            val cnnPredPaddle = NeuralModels.runProbeDecoder(acts.cnnConv1, "paddleX")
            val dtPredPaddle = NeuralModels.runProbeDecoder(acts.dtEmbed, "paddleX")
            
            val cnnPredBall = NeuralModels.runProbeDecoder(acts.cnnDense, "ballX")
            val dtPredBall = NeuralModels.runProbeDecoder(acts.dtEmbed, "ballX")

            cnnPaddleErrorSum += (groundPaddle - cnnPredPaddle).pow(2)
            dtPaddleErrorSum += (groundPaddle - dtPredPaddle).pow(2)
            cnnBallErrorSum += (groundBallX - cnnPredBall).pow(2)
            dtBallErrorSum += (groundBallX - dtPredBall).pow(2)
        }

        return mapOf(
            "cnn_paddle_rmse" to sqrt(cnnPaddleErrorSum / n),
            "dt_paddle_rmse" to sqrt(dtPaddleErrorSum / n),
            "cnn_ball_rmse" to sqrt(cnnBallErrorSum / n),
            "dt_ball_rmse" to sqrt(dtBallErrorSum / n)
        )
    }
}
