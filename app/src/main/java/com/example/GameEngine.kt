package com.example

import kotlin.random.Random

data class Brick(
    val id: Int,
    val row: Int,
    val col: Int,
    var isDestroyed: Boolean,
    val scoreValue: Int = 10
)

data class SequenceStep(
    val state: FloatArray, // [paddleX, ballX, ballY] (normalized)
    val action: Int,       // 0=Left, 1=Right, 2=Stay
    val rtg: Float         // Return-to-go
)

class GameEngine {
    var state = GameState()
    var frameHistory = mutableListOf<SequenceStep>()
    var maxHistorySize = 5
    
    // Grid measurements
    val areaWidth = 100f
    val areaHeight = 100f
    val brickRows = 4
    val brickCols = 8
    val brickWidth = 12.5f // 100 / 8
    val brickHeight = 5f
    val brickTopOffset = 15f
    
    // Game statistics
    var totalSteps = 0
    
    init {
        resetGame()
    }
    
    fun resetGame() {
        state = GameState(
            ballX = 50f,
            ballY = 50f,
            ballVx = 1.5f,
            ballVy = 1.5f,
            paddleX = 40f,
            paddleWidth = 20f,
            bricks = createInitialBricks(),
            score = 0,
            lives = 3,
            isGameOver = false
        )
        frameHistory.clear()
        // Initialize history with padding steps
        for (i in 0 until maxHistorySize) {
            frameHistory.add(SequenceStep(floatArrayOf(0.4f, 0.5f, 0.5f), 2, 1.0f))
        }
        totalSteps = 0
    }
    
    private fun createInitialBricks(): List<Brick> {
        val list = mutableListOf<Brick>()
        var id = 0
        for (r in 0 until brickRows) {
            for (c in 0 until brickCols) {
                list.add(Brick(id = id++, row = r, col = c, isDestroyed = false))
            }
        }
        return list
    }
    
    // Interaction: Toggle brick state at visual tap location
    fun handleScreenTouch(tx: Float, ty: Float) {
        if (ty >= brickTopOffset && ty <= brickTopOffset + (brickRows * brickHeight)) {
            val col = (tx / brickWidth).toInt().coerceIn(0, brickCols - 1)
            val row = ((ty - brickTopOffset) / brickHeight).toInt().coerceIn(0, brickRows - 1)
            state.bricks.find { it.row == row && it.col == col }?.let { brick ->
                brick.isDestroyed = !brick.isDestroyed
            }
        } else {
            // Relocate ball
            state.ballX = tx.coerceIn(5f, 95f)
            state.ballY = ty.coerceIn(10f, 85f)
            // Gently nudge speed towards paddle
            state.ballVy = kotlin.math.abs(state.ballVy)
        }
    }
    
    // Updates physical step of game
    fun step(action: Int) {
        if (state.isGameOver) return
        
        state.lastAction = action
        totalSteps++
        
        // Move paddle
        val paddleSpeed = 3.0f
        when (action) {
            0 -> state.paddleX = (state.paddleX - paddleSpeed).coerceAtLeast(0f)
            1 -> state.paddleX = (state.paddleX + paddleSpeed).coerceAtMost(100f - state.paddleWidth)
        }
        
        // Move ball
        state.ballX += state.ballVx
        state.ballY += state.ballVy
        
        // Wall collisions
        if (state.ballX <= 0f) {
            state.ballX = 0f
            state.ballVx = kotlin.math.abs(state.ballVx)
        } else if (state.ballX >= areaWidth) {
            state.ballX = areaWidth
            state.ballVx = -kotlin.math.abs(state.ballVx)
        }
        
        if (state.ballY <= 0f) {
            state.ballY = 0f
            state.ballVy = kotlin.math.abs(state.ballVy)
        }
        
        // Paddle collision
        val paddleY = 92f
        if (state.ballY >= paddleY && state.ballY <= paddleY + 3f) {
            val paddleLeft = state.paddleX
            val paddleRight = state.paddleX + state.paddleWidth
            if (state.ballX >= paddleLeft && state.ballX <= paddleRight) {
                // Bounce ball
                state.ballY = paddleY
                state.ballVy = -kotlin.math.abs(state.ballVy)
                // Add spin based on where it hit paddle (relative offset from center)
                val paddleCenter = paddleLeft + (state.paddleWidth / 2)
                val relativeHit = (state.ballX - paddleCenter) / (state.paddleWidth / 2)
                state.ballVx = (state.ballVx + relativeHit * 0.8f).coerceIn(-3.0f, 3.0f)
            }
        }
        
        // Ball out of bounds (bottom)
        if (state.ballY >= areaHeight) {
            state.lives--
            if (state.lives <= 0) {
                state.isGameOver = true
            } else {
                // Reset ball
                state.ballX = 50f
                state.ballY = 50f
                state.ballVx = if (Random.nextBoolean()) 1.5f else -1.5f
                state.ballVy = -1.5f
            }
        }
        
        // Brick collision
        for (brick in state.bricks) {
            if (!brick.isDestroyed) {
                val bLeft = brick.col * brickWidth
                val bRight = bLeft + brickWidth
                val bTop = brickTopOffset + (brick.row * brickHeight)
                val bBottom = bTop + brickHeight
                
                // Box collision check
                if (state.ballX >= bLeft && state.ballX <= bRight &&
                    state.ballY >= bTop && state.ballY <= bBottom) {
                    
                    brick.isDestroyed = true
                    state.score += brick.scoreValue
                    
                    // Bounce path
                    val overlapLeft = state.ballX - bLeft
                    val overlapRight = bRight - state.ballX
                    val overlapTop = state.ballY - bTop
                    val overlapBottom = bBottom - state.ballY
                    
                    val minOverlapHorizontal = kotlin.math.min(overlapLeft, overlapRight)
                    val minOverlapVertical = kotlin.math.min(overlapTop, overlapBottom)
                    
                    if (minOverlapHorizontal < minOverlapVertical) {
                        state.ballVx = if (overlapLeft < overlapRight) -kotlin.math.abs(state.ballVx) else kotlin.math.abs(state.ballVx)
                    } else {
                        state.ballVy = if (overlapTop < overlapBottom) -kotlin.math.abs(state.ballVy) else kotlin.math.abs(state.ballVy)
                    }
                    break // Collide with at most one brick per frame
                }
            }
        }
        
        // Append sequence frame
        val currentS = floatArrayOf(
            state.paddleX / areaWidth,
            state.ballX / areaWidth,
            state.ballY / areaHeight
        )
        // RTG based on target score 200 max
        val currentRTG = (200f - state.score).coerceAtLeast(0f) / 200f
        
        frameHistory.add(SequenceStep(currentS, action, currentRTG))
        if (frameHistory.size > maxHistorySize) {
            frameHistory.removeAt(0)
        }
    }
    
    // Representation builders
    
    // 1) CNN Vision Input: 16x16 pixel normalized floats
    fun getCnnVisionInput(): Array<FloatArray> {
        val grid = Array(16) { FloatArray(16) { 0f } }
        
        // Draw bricks: block 0.4
        for (brick in state.bricks) {
            if (!brick.isDestroyed) {
                val gridRow = (((brickTopOffset + brick.row * brickHeight) / areaHeight) * 16).toInt().coerceIn(0, 15)
                val gridColStart = (((brick.col * brickWidth) / areaWidth) * 16).toInt().coerceIn(0, 15)
                val gridColEnd = ((((brick.col + 1) * brickWidth) / areaWidth) * 16).toInt().coerceIn(0, 15)
                for (c in gridColStart..gridColEnd) {
                    grid[gridRow][c] = 0.4f
                }
            }
        }
        
        // Draw paddle: block 0.7
        val padRow = ((92f / areaHeight) * 16).toInt().coerceIn(0, 15)
        val padColStart = (((state.paddleX) / areaWidth) * 16).toInt().coerceIn(0, 15)
        val padColEnd = (((state.paddleX + state.paddleWidth) / areaWidth) * 16).toInt().coerceIn(0, 15)
        for (c in padColStart..padColEnd) {
            grid[padRow][c] = 0.7f
        }
        
        // Draw ball: block 1.0
        val bRow = ((state.ballY / areaHeight) * 16).toInt().coerceIn(0, 15)
        val bCol = ((state.ballX / areaWidth) * 16).toInt().coerceIn(0, 15)
        grid[bRow][bCol] = 1.0f
        
        return grid
    }
    
    // 2) DQN RAM Input: vector of 6 sensory floats
    fun getDqnRamInput(): FloatArray {
        val activeBricks = state.bricks.count { !it.isDestroyed }
        return floatArrayOf(
            state.paddleX / areaWidth,
            state.ballX / areaWidth,
            state.ballY / areaHeight,
            (state.ballVx + 3.0f) / 6.0f, // shift to [0,1]
            (state.ballVy + 3.0f) / 6.0f,
            activeBricks.toFloat() / (brickRows * brickCols)
        )
    }
    
    // 3) DT Sequence Input
    fun getDtTrajectoryInput(): List<SequenceStep> {
        return frameHistory.toList()
    }
}

data class GameState(
    var ballX: Float = 50f,
    var ballY: Float = 50f,
    var ballVx: Float = 1.5f,
    var ballVy: Float = 1.5f,
    var paddleX: Float = 40f,
    var paddleWidth: Float = 20f,
    val bricks: List<Brick> = emptyList(),
    var score: Int = 0,
    var lives: Int = 3,
    var lastAction: Int = 2,
    var isGameOver: Boolean = false
)
