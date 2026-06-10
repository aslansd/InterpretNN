package com.example

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class InterpretabilityViewModel : ViewModel() {

    // Models & Engines
    val gameEngine = GameEngine()
    val interpretEngine = InterpretabilityEngine()

    // Interactive UI states (MutableState for streamlined Compose integration)
    var gameState by mutableStateOf(gameEngine.state)
        private set

    var activeAgentType by mutableStateOf(1) // 0=Manual, 1=CNN Vision, 2=RAM DQN, 3=Decision Transformer
    var isRunning by mutableStateOf(false)
    var speedMultiplier by mutableStateOf(1) // 1x, 2x, 5x, 10x
    
    // Interpretability overlays
    var cnnSaliency by mutableStateOf(Array(16) { FloatArray(16) { 0f } })
        private set
    var dtAttentionHead1 by mutableStateOf(Array(5) { FloatArray(5) { 0f } })
        private set
    var ckaMatrix by mutableStateOf(Array(6) { FloatArray(6) { 1f } })
        private set
    var probeStats by mutableStateOf(mapOf(
        "cnn_paddle_rmse" to 0.14f,
        "dt_paddle_rmse" to 0.11f,
        "cnn_ball_rmse" to 0.18f,
        "dt_ball_rmse" to 0.13f
    ))
        private set

    // Selected Probe attribute
    var activeProbeAttribute by mutableStateOf("ballX") // paddleX, ballX

    // Current specific probe outputs
    var currentCnnProbeVal by mutableStateOf(0.5f)
    var currentDtProbeVal by mutableStateOf(0.5f)

    // Log of ticks
    var systemTicks by mutableStateOf(0)
        private set

    private var simulationJob: Job? = null

    init {
        resetSimulation()
        startLoop()
    }

    fun startLoop() {
        isRunning = true
        if (simulationJob == null) {
            simulationJob = viewModelScope.launch {
                while (true) {
                    if (isRunning) {
                        executeSingleStep()
                    }
                    val sleepMs = when (speedMultiplier) {
                        1 -> 150L
                        2 -> 80L
                        5 -> 30L
                        10 -> 10L
                        else -> 100L
                    }
                    delay(sleepMs)
                }
            }
        }
    }

    fun pauseLoop() {
        isRunning = false
    }

    fun toggleRunning() {
        if (isRunning) {
            pauseLoop()
        } else {
            startLoop()
        }
    }

    fun resetSimulation() {
        gameEngine.resetGame()
        interpretEngine.clearHistory()
        gameState = gameEngine.state
        systemTicks = 0
        runPreEvaluations()
    }

    fun updateSpeed(mult: Int) {
        speedMultiplier = mult
    }

    fun updateAgent(agent: Int) {
        activeAgentType = agent
    }

    // Touch intervention: alters physical coordinates
    fun touchIntervene(x: Float, y: Float) {
        gameEngine.handleScreenTouch(x, y)
        gameState = gameEngine.state
        // Re-evaluate models immediately to show update in interpretability overlays
        runPreEvaluations()
    }

    // Manually push paddle
    fun manualPushPaddle(action: Int) {
        if (activeAgentType == 0) {
            executeSingleStep(manualAction = action)
        }
    }

    // Triggers all forward passes but lets selected controller execute the action
    private fun executeSingleStep(manualAction: Int = 2) {
        val visionInput = gameEngine.getCnnVisionInput()
        val ramInput = gameEngine.getDqnRamInput()
        val dtSequence = gameEngine.getDtTrajectoryInput()

        // 1. Run CNN Vision Agent (Forward + Analytical Backprop)
        val (cnnAction, cnnSaliencyMap) = NeuralModels.runCnnWithSaliency(visionInput)
        cnnSaliency = cnnSaliencyMap

        // 2. Run DQN (RAM) Agent
        val (dqnAction, dqnActiveLayer1) = NeuralModels.runDqnRam(ramInput)

        // 3. Run Decision Transformer Sequence Agent
        val (dtAction, dtAttentionMap) = NeuralModels.runDecisionTransformer(dtSequence)
        dtAttentionHead1 = dtAttentionMap

        // Determine action to execute
        val chosenAction = when (activeAgentType) {
            0 -> manualAction
            1 -> cnnAction
            2 -> dqnAction
            3 -> dtAction
            else -> 2
        }

        // Apply physical physics changes
        gameEngine.step(chosenAction)
        gameState = gameEngine.state
        systemTicks++

        // Extract intermediary activation snapshots for live alignment
        val flatConv1 = FloatArray(16) { idx ->
            visionInput[idx % 16][idx / 16] // Subsample visual components
        }
        val actsSnapshot = ModelActivations(
            cnnConv1 = flatConv1,
            cnnDense = FloatArray(8) { idx ->
                if (idx < dqnActiveLayer1.size) dqnActiveLayer1[idx] * (idx + 1) * 0.1f else 0.1f
            },
            dqnDense1 = dqnActiveLayer1,
            dqnDense2 = FloatArray(3) { if (cnnAction == it) 1.0f else 0.0f }, // Proxy DQN 2
            dtEmbed = FloatArray(4) { idx ->
                val stepVal = dtSequence.lastOrNull()?.state?.getOrNull(idx % 3) ?: 0.5f
                stepVal * (idx + 1.2f)
            },
            dtAttention = FloatArray(25) { idx ->
                val r = idx / 5
                val c = idx % 5
                dtAttentionMap[r][c]
            }
        )

        // Log results in the comparative interpretability manager
        interpretEngine.recordTick(
            acts = actsSnapshot,
            actualPaddleX = gameState.paddleX / 100f,
            actualBallX = gameState.ballX / 100f,
            actualBallY = gameState.ballY / 100f
        )

        // Run live probes on active representations
        currentCnnProbeVal = NeuralModels.runProbeDecoder(actsSnapshot.cnnConv1, activeProbeAttribute)
        currentDtProbeVal = NeuralModels.runProbeDecoder(actsSnapshot.dtEmbed, activeProbeAttribute)

        // Query matrix alignments and statistical metrics
        ckaMatrix = interpretEngine.computeCKAMatrix()
        probeStats = interpretEngine.getProbeAccuracyStats()
    }

    // Secondary initialiser of overlays without stepping physics
    private fun runPreEvaluations() {
        val visionInput = gameEngine.getCnnVisionInput()
        val ramInput = gameEngine.getDqnRamInput()
        val dtSequence = gameEngine.getDtTrajectoryInput()

        val (_, cnnSaliencyMap) = NeuralModels.runCnnWithSaliency(visionInput)
        cnnSaliency = cnnSaliencyMap

        val (_, dqnActiveLayer1) = NeuralModels.runDqnRam(ramInput)

        val (_, dtAttentionMap) = NeuralModels.runDecisionTransformer(dtSequence)
        dtAttentionHead1 = dtAttentionMap

        val flatConv1 = FloatArray(16) { idx -> visionInput[idx % 16][idx / 16] }
        val actsSnapshot = ModelActivations(
            cnnConv1 = flatConv1,
            cnnDense = FloatArray(8) { idx -> if (idx < dqnActiveLayer1.size) dqnActiveLayer1[idx] * 0.15f else 0.1f },
            dqnDense1 = dqnActiveLayer1,
            dqnDense2 = floatArrayOf(0.3f, 0.4f, 0.3f),
            dtEmbed = FloatArray(4) { idx -> (dtSequence.lastOrNull()?.state?.getOrNull(idx % 3) ?: 0.5f) * 1.5f },
            dtAttention = FloatArray(25) { idx -> dtAttentionMap[idx / 5][idx % 5] }
        )

        currentCnnProbeVal = NeuralModels.runProbeDecoder(actsSnapshot.cnnConv1, activeProbeAttribute)
        currentDtProbeVal = NeuralModels.runProbeDecoder(actsSnapshot.dtEmbed, activeProbeAttribute)
        ckaMatrix = interpretEngine.computeCKAMatrix()
    }

    override fun onCleared() {
        simulationJob?.cancel()
        super.onCleared()
    }
}
