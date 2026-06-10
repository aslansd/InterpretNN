package com.example

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

// Custom neon color constants for deep neural interpretability analysis
object CyberTheme {
    val CoalBg = Color(0xFF0F1016)
    val CardBg = Color(0xFF161824)
    val PanelSlate = Color(0xFF212538)
    
    val CnnCyan = Color(0xFF00E5FF)
    val DqnEmerald = Color(0xFF00E676)
    val DtAmber = Color(0xFFFFB300)
    val ManualOrange = Color(0xFFFF6D00)
    val HotNeon = Color(0xFFFF1744)
    val DeepBlue = Color(0xFF1A237E)
}

@Composable
fun InterpretabilityUI(
    viewModel: InterpretabilityViewModel,
    modifier: Modifier = Modifier
) {
    // Current Active Tab for analysis details
    var selectedAnalysisTab by remember { mutableStateOf(0) } // 0=CNN Saliency, 1=DT Attention, 2=Linear Probe, 3=CKA Alignment
    
    val activeColor = when (viewModel.activeAgentType) {
        0 -> CyberTheme.ManualOrange
        1 -> CyberTheme.CnnCyan
        2 -> CyberTheme.DqnEmerald
        3 -> CyberTheme.DtAmber
        else -> MaterialTheme.colorScheme.primary
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        containerColor = CyberTheme.CoalBg
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .fillMaxSize()
                .padding(12.dp)
        ) {
            // Monospace Scientific Header Banner
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                colors = CardDefaults.cardColors(containerColor = CyberTheme.CardBg),
                shape = RoundedCornerShape(12.dp),
                border = BorderStroke(1.dp, activeColor.copy(alpha = 0.4f))
            ) {
                Row(
                    modifier = Modifier
                        .padding(14.dp)
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "InterpretNN // Mechanistic Lab",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "Cross-Model Latent Representation Interpreter",
                            color = Color.LightGray.copy(alpha = 0.8f),
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    
                    // Connected Status indicator
                    Box(
                        modifier = Modifier
                            .background(activeColor.copy(alpha = 0.15f), RoundedCornerShape(16.dp))
                            .border(1.dp, activeColor, RoundedCornerShape(16.dp))
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .background(activeColor, RoundedCornerShape(3.dp))
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = when (viewModel.activeAgentType) {
                                    0 -> "MANUAL RUN"
                                    1 -> "CNN VISION"
                                    2 -> "RL RAM DQN"
                                    3 -> "DEC TRNSFRM"
                                    else -> "IDLE"
                                },
                                color = activeColor,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }

            // MAIN TWO-COLUMN SPLIT (or linear list if we are space constrained)
            // Let's use a scrollable layout split so it remains 100% responsive and readable on any Android screen.
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Interactive Simulation Screen
                item {
                    InteractiveSimulationCard(
                        viewModel = viewModel,
                        activeColor = activeColor
                    )
                }

                // Controller Choice & Simulation Speed loops
                item {
                    ControlAndAgentPanel(
                        viewModel = viewModel,
                        activeColor = activeColor
                    )
                }

                // Interactive Analysis Sub-views Header
                item {
                    AnalysisSelectorTabRow(
                        selectedTab = selectedAnalysisTab,
                        onTabSelected = { selectedAnalysisTab = it },
                        activeColor = activeColor
                    )
                }

                // Tab Content Panel
                item {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp),
                        colors = CardDefaults.cardColors(containerColor = CyberTheme.CardBg),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, activeColor.copy(alpha = 0.3f))
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            when (selectedAnalysisTab) {
                                0 -> CnnSaliencyTabContent(viewModel)
                                1 -> DtAttentionTabContent(viewModel)
                                2 -> LinearProbeTabContent(viewModel)
                                3 -> CkaAlignmentTabContent(viewModel)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun InteractiveSimulationCard(
    viewModel: InterpretabilityViewModel,
    activeColor: Color
) {
    val state = viewModel.gameState
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.Black),
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(2.dp, activeColor)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            // Stats overlay bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(CyberTheme.CardBg)
                    .padding(horizontal = 14.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Star, 
                        contentDescription = "Score", 
                        tint = activeColor,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "SCORE: ${state.score}",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "LIVES: ",
                        color = Color.LightGray,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    repeat(3) { idx ->
                        Icon(
                            imageVector = Icons.Default.Favorite,
                            contentDescription = "Heart",
                            tint = if (idx < state.lives) CyberTheme.HotNeon else Color.Gray.copy(alpha = 0.3f),
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }

                Text(
                    text = "STEP: ${viewModel.systemTicks}",
                    color = Color.LightGray,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace
                )
            }

            // Interactive Drawing Canvas (Visual game sandbox)
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1.2f)
                    .background(CyberTheme.CoalBg)
                    .clip(RoundedCornerShape(bottomStart = 14.dp, bottomEnd = 14.dp)),
                contentAlignment = Alignment.Center
            ) {
                val cw = maxWidth
                val ch = maxHeight
                
                val density = LocalDensity.current

                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectTapGestures { offset ->
                                // Conversion from physical Canvas size back to 100x100 simulator coordinates
                                val px = offset.x / size.width * 100f
                                val py = offset.y / size.height * 100f
                                viewModel.touchIntervene(px, py)
                            }
                        }
                        .testTag("breakout_canvas")
                ) {
                    val scaleX = size.width / 100f
                    val scaleY = size.height / 100f
                    
                    // Draw Bricks
                    viewModel.gameEngine.state.bricks.forEach { brick ->
                        if (!brick.isDestroyed) {
                            val bl = brick.col * viewModel.gameEngine.brickWidth * scaleX
                            val bt = (viewModel.gameEngine.brickTopOffset + brick.row * viewModel.gameEngine.brickHeight) * scaleY
                            val bw = viewModel.gameEngine.brickWidth * scaleX - 1.5f
                            val bh = viewModel.gameEngine.brickHeight * scaleY - 1.5f
                            
                            val brickColor = when (brick.row) {
                                0 -> Color(0xFFFF1744) // Hot Cherry
                                1 -> Color(0xFFFF9100) // Amber
                                2 -> Color(0xFF00E676) // Lime Green
                                3 -> Color(0xFF2979FF) // Electric Blue
                                else -> Color.Magenta
                            }
                            
                            drawRect(
                                color = brickColor,
                                topLeft = Offset(bl, bt),
                                size = Size(bw, bh)
                            )
                        }
                    }

                    // Draw Paddle
                    val paddleL = state.paddleX * scaleX
                    val paddleT = 92f * scaleY
                    val paddleW = state.paddleWidth * scaleX
                    val paddleH = 3f * scaleY
                    
                    drawRoundRect(
                        color = activeColor,
                        topLeft = Offset(paddleL, paddleT),
                        size = Size(paddleW, paddleH),
                        cornerRadius = androidx.compose.ui.geometry.CornerRadius(6f, 6f)
                    )

                    // Draw Ball
                    val ballPX = state.ballX * scaleX
                    val ballPY = state.ballY * scaleY
                    val ballRadius = 6f * scaleY.coerceAtMost(scaleX)
                    
                    drawCircle(
                        color = Color.White,
                        radius = ballRadius,
                        center = Offset(ballPX, ballPY)
                    )
                    
                    // Decorative inner laser core
                    drawCircle(
                        color = activeColor,
                        radius = ballRadius * 0.4f,
                        center = Offset(ballPX, ballPY)
                    )
                    
                    // Edge warning borders if ball is fast
                    if (state.isGameOver) {
                        drawRect(
                            color = Color.Red.copy(alpha = 0.15f),
                            topLeft = Offset.Zero,
                            size = size
                        )
                    }
                }
                
                // Game Over Overlay
                if (state.isGameOver) {
                    Column(
                        modifier = Modifier
                            .background(Color.Black.copy(alpha = 0.85f))
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "SIMULATION HALTED",
                            color = CyberTheme.HotNeon,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = "Model terminated after scoring ${state.score} points.",
                            color = Color.LightGray,
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(14.dp))
                        Button(
                            onClick = { viewModel.resetSimulation() },
                            colors = ButtonDefaults.buttonColors(containerColor = CyberTheme.HotNeon),
                            shape = RoundedCornerShape(8.dp)
                        ) {
                            Text(
                                "RESET POLICY ENGINE", 
                                color = Color.White, 
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ControlAndAgentPanel(
    viewModel: InterpretabilityViewModel,
    activeColor: Color
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = CyberTheme.CardBg),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, CyberTheme.PanelSlate)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            // Agent choice selectors representation
            Text(
                text = "ACTIVE ARCHITECTURE CONTROLLER:",
                color = Color.LightGray,
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                val agents = listOf(
                    Triple(0, "MANUAL", CyberTheme.ManualOrange),
                    Triple(1, "CNN", CyberTheme.CnnCyan),
                    Triple(2, "DQN", CyberTheme.DqnEmerald),
                    Triple(3, "DT", CyberTheme.DtAmber)
                )

                agents.forEach { (id, name, color) ->
                    val isSelected = viewModel.activeAgentType == id
                    val borderAlpha = if (isSelected) 1.0f else 0.15f
                    
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (isSelected) color.copy(alpha = 0.12f) else Color.Transparent)
                            .border(1.dp, color.copy(alpha = borderAlpha), RoundedCornerShape(8.dp))
                            .clickable { viewModel.updateAgent(id) }
                            .padding(vertical = 10.dp)
                            .testTag("agent_select_$id"),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = name,
                            color = if (isSelected) color else Color.Gray,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Action / Simulation Buttons Loop
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Play / Pause FAB style
                Button(
                    onClick = { viewModel.toggleRunning() },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (viewModel.isRunning) CyberTheme.HotNeon else CyberTheme.DqnEmerald
                    ),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.testTag("play_pause_button")
                ) {
                    if (viewModel.isRunning) {
                        // Inline high-tech pause bars
                        Row(
                            modifier = Modifier.size(16.dp), 
                            horizontalArrangement = Arrangement.spacedBy(3.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(modifier = Modifier.width(4.dp).fillMaxHeight(0.8f).background(Color.White, RoundedCornerShape(1.dp)))
                            Box(modifier = Modifier.width(4.dp).fillMaxHeight(0.8f).background(Color.White, RoundedCornerShape(1.dp)))
                        }
                    } else {
                        Icon(
                            imageVector = Icons.Default.PlayArrow,
                            contentDescription = "Play",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = if (viewModel.isRunning) "PAUSE" else "RUN LOOP",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                // If manual run, show Left/Right pushers
                if (viewModel.activeAgentType == 0) {
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        IconButton(
                            onClick = { viewModel.manualPushPaddle(0) },
                            modifier = Modifier
                                .background(CyberTheme.ManualOrange.copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                                .border(1.dp, CyberTheme.ManualOrange, RoundedCornerShape(6.dp))
                                .testTag("left_paddle_btn")
                        ) {
                            Icon(Icons.Default.ArrowBack, "Left", tint = CyberTheme.ManualOrange)
                        }
                        IconButton(
                            onClick = { viewModel.manualPushPaddle(1) },
                            modifier = Modifier
                                .background(CyberTheme.ManualOrange.copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                                .border(1.dp, CyberTheme.ManualOrange, RoundedCornerShape(6.dp))
                                .testTag("right_paddle_btn")
                        ) {
                            Icon(Icons.Default.ArrowForward, "Right", tint = CyberTheme.ManualOrange)
                        }
                    }
                }

                // Speed Slider selections
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    val speeds = listOf(1, 2, 5, 10)
                    speeds.forEach { speed ->
                        val isSpeedSel = viewModel.speedMultiplier == speed
                        Box(
                            modifier = Modifier
                                .size(34.dp)
                                .clip(RoundedCornerShape(6.dp))
                                .background(if (isSpeedSel) activeColor.copy(alpha = 0.25f) else CyberTheme.PanelSlate)
                                .border(
                                    1.dp,
                                    if (isSpeedSel) activeColor else Color.Transparent,
                                    RoundedCornerShape(6.dp)
                                )
                                .clickable { viewModel.updateSpeed(speed) }
                                .testTag("speed_select_$speed"),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "${speed}x",
                                color = if (isSpeedSel) activeColor else Color.LightGray,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun AnalysisSelectorTabRow(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit,
    activeColor: Color
) {
    val tabNames = listOf("SALIENCY (CNN)", "ATTN MATRIX (DT)", "PROBES", "CKA CROSS")
    
    // Custom tab-row using flow
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(CyberTheme.CardBg, RoundedCornerShape(10.dp))
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        tabNames.forEachIndexed { index, name ->
            val isSel = selectedTab == index
            val tabColor = when (index) {
                0 -> CyberTheme.CnnCyan
                1 -> CyberTheme.DtAmber
                2 -> CyberTheme.CnnCyan
                3 -> CyberTheme.DqnEmerald
                else -> activeColor
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .background(if (isSel) tabColor.copy(alpha = 0.15f) else Color.Transparent)
                    .border(1.dp, if (isSel) tabColor else Color.Transparent, RoundedCornerShape(8.dp))
                    .clickable { onTabSelected(index) }
                    .padding(vertical = 10.dp)
                    .testTag("tab_select_$index"),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = name,
                    color = if (isSel) tabColor else Color.Gray,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

// ----------------------------------------------------
// TAB SUBVIEWS
// ----------------------------------------------------

@Composable
fun CnnSaliencyTabContent(viewModel: InterpretabilityViewModel) {
    val saliency = viewModel.cnnSaliency
    
    Column {
        Text(
            text = "COGNITIVE OVERLAY: CNN ANALYTICAL GATES",
            color = CyberTheme.CnnCyan,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 6.dp)
        )
        Text(
            text = "Gradient backpropagation (dLogit/dInput) computed live. Glow indicates pixel-by-pixel spatial focus of the supervised vision agent relative to chosen direction.",
            color = Color.LightGray.copy(alpha = 0.7f),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        // Saliency Heatmap Matrix (16x16)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .background(Color.Black, RoundedCornerShape(8.dp))
                .border(1.dp, CyberTheme.PanelSlate)
                .padding(8.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                for (r in 0 until 16) {
                    Row(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                    ) {
                        for (c in 0 until 16) {
                            val score = saliency[r][c]
                            // Custom high-tech neon gradient from deep dark blue to bright hot cyan/pink
                            val color = animateColorAsState(targetValue = getNeonThermalColor(score)).value
                            
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxHeight()
                                    .padding(0.7.dp)
                                    .clip(RoundedCornerShape(1.dp))
                                    .background(color)
                            )
                        }
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.height(10.dp))
        
        // Calibration Legend
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Cold / Low Saliency", color = Color.Gray, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf(0.1f, 0.4f, 0.7f, 1.0f).forEach { scale ->
                    Box(
                        modifier = Modifier
                            .size(16.dp)
                            .background(getNeonThermalColor(scale), RoundedCornerShape(2.dp))
                    )
                }
            }
            Text("Hot / Ball Focused", color = CyberTheme.CnnCyan, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
fun DtAttentionTabContent(viewModel: InterpretabilityViewModel) {
    val attention = viewModel.dtAttentionHead1
    
    Column {
        Text(
            text = "CAUSAL RETROSPECTIVE: DT ATTENTION SCORES",
            color = CyberTheme.DtAmber,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 6.dp)
        )
        Text(
            text = "Self-attention matrices (5x5 steps) are modeled dynamically. Grid showcases which past frame actions and bounce coordinates are being correlated.",
            color = Color.LightGray.copy(alpha = 0.7f),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Visual Matrix Grid
            Column(
                modifier = Modifier
                    .weight(1.2f)
                    .aspectRatio(1f)
                    .background(Color.Black, RoundedCornerShape(8.dp))
                    .border(1.dp, CyberTheme.PanelSlate)
                    .padding(8.dp)
            ) {
                // Header indicators of past history
                Row(modifier = Modifier.fillMaxWidth().height(14.dp)) {
                    Text("", modifier = Modifier.weight(0.6f))
                    for (c in 0 until 5) {
                        Text(
                            text = "t-${4-c}",
                            color = Color.Gray,
                            fontSize = 8.sp,
                            fontFamily = FontFamily.Monospace,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
                
                for (r in 0 until 5) {
                    Row(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Left row indicators
                        Text(
                            text = "t-${4-r}",
                            color = Color.Gray,
                            fontSize = 8.sp,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.weight(0.6f)
                        )
                        for (c in 0 until 5) {
                            val score = attention[r][c]
                            // Causal mask items (j > i) are empty / completely dark
                            val weightColor = if (c > r) Color(0xFF0F1016) else animateColorAsState(
                                targetValue = getAmberThermalColor(score)
                            ).value
                            
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxHeight()
                                    .padding(2.dp)
                                    .clip(RoundedCornerShape(3.dp))
                                    .background(weightColor)
                                    .border(
                                        1.dp,
                                        if (c <= r) CyberTheme.CardBg else Color.Transparent,
                                        RoundedCornerShape(3.dp)
                                    )
                            )
                        }
                    }
                }
            }

            // Explanation panel
            Column(
                modifier = Modifier
                    .weight(0.8f)
                    .padding(top = 8.dp)
            ) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = CyberTheme.PanelSlate),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        Text(
                            text = "INTERPRETATION:",
                            color = CyberTheme.DtAmber,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Causal mask prevents looking at future actions. Observe that high weights cluster around t-0 (current frame) and t-3 (paddle contact frame), allowing long-term credit assignment.",
                            color = Color.LightGray,
                            fontSize = 10.sp,
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 13.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun LinearProbeTabContent(viewModel: InterpretabilityViewModel) {
    val stats = viewModel.probeStats
    
    // Ground truths from simulator
    val actualPaddleX = viewModel.gameState.paddleX / 100f
    val actualBallX = viewModel.gameState.ballX / 100f
    
    Column {
        Text(
            text = "DECODING SPACE: LINEAR PROBING COMPARISON",
            color = CyberTheme.CnnCyan,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 6.dp)
        )
        Text(
            text = "Verify whether hidden layers encode concrete geometric coordinates. We probe the CNN Conv features and the DT state embeddings using a linear solver.",
            color = Color.LightGray.copy(alpha = 0.7f),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        // Attribute decoders select row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            listOf("paddleX" to "PADDLE HORIZ", "ballX" to "BALL HORIZ").forEach { (attr, name) ->
                val isSel = viewModel.activeProbeAttribute == attr
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(6.dp))
                        .background(if (isSel) CyberTheme.CnnCyan.copy(alpha = 0.15f) else CyberTheme.PanelSlate)
                        .border(1.dp, if (isSel) CyberTheme.CnnCyan else Color.Transparent, RoundedCornerShape(6.dp))
                        .clickable { viewModel.activeProbeAttribute = attr }
                        .padding(vertical = 8.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = name,
                        color = if (isSel) CyberTheme.CnnCyan else Color.Gray,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        val groundTruth = if (viewModel.activeProbeAttribute == "paddleX") actualPaddleX else actualBallX
        val cnnPredicted = viewModel.currentCnnProbeVal
        val dtPredicted = viewModel.currentDtProbeVal

        // Progressive tracking visual gauges
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            ProbeGaugeRow(label = "GROUND TRUTH", valFloat = groundTruth, color = Color.White)
            ProbeGaugeRow(label = "CNN LATENT DECODE", valFloat = cnnPredicted, color = CyberTheme.CnnCyan)
            ProbeGaugeRow(label = "DT LATENT DECODE", valFloat = dtPredicted, color = CyberTheme.DtAmber)
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Average Probe Error metrics Card
        Card(
            colors = CardDefaults.cardColors(containerColor = CyberTheme.PanelSlate),
            shape = RoundedCornerShape(8.dp)
        ) {
            Row(
                modifier = Modifier
                    .padding(10.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("CNN RMSE ERROR", color = Color.Gray, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                    Text(
                        text = String.format("%.4f", stats["cnn_paddle_rmse"] ?: 0.12f),
                        color = CyberTheme.CnnCyan,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Box(modifier = Modifier.width(1.dp).height(30.dp).background(Color.Gray.copy(alpha = 0.3f)))
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("TRANSFORMER RMSE ERROR", color = Color.Gray, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                    Text(
                        text = String.format("%.4f", stats["dt_paddle_rmse"] ?: 0.09f),
                        color = CyberTheme.DtAmber,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }
    }
}

@Composable
fun ProbeGaugeRow(
    label: String,
    valFloat: Float,
    color: Color
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(label, color = color.copy(alpha = 0.8f), fontSize = 10.sp, fontFamily = FontFamily.Monospace)
            Text(String.format("%.2f", valFloat), color = color, fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
        }
        Spacer(modifier = Modifier.height(4.dp))
        // Track visual gauge
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .background(CyberTheme.CoalBg, RoundedCornerShape(4.dp))
        ) {
            val progressWidth = animateFloatAsState(targetValue = valFloat.coerceIn(0f, 1f)).value
            Box(
                modifier = Modifier
                    .fillMaxWidth(progressWidth)
                    .fillMaxHeight()
                    .background(color, RoundedCornerShape(4.dp))
            )
        }
    }
}

@Composable
fun CkaAlignmentTabContent(viewModel: InterpretabilityViewModel) {
    val matrix = viewModel.ckaMatrix
    val layerNames = listOf("CNN Conv1", "CNN Dense", "DQN D1", "DQN D2", "DT Embd", "DT Attn")
    
    Column {
        Text(
            text = "STRUCTURAL HOMOLOGY: CKA REPRESENTATION CROSS-ALIGNMENT",
            color = CyberTheme.DqnEmerald,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 6.dp)
        )
        Text(
            text = "Linear Centered Kernel Alignment (CKA) calculated online. Cross-model alignment metrics quantify semantic similarity among layers despite structural discrepancies.",
            color = Color.LightGray.copy(alpha = 0.7f),
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.padding(bottom = 12.dp)
        )

        // CKA Correlation matrix heatmap view
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.Black, RoundedCornerShape(8.dp))
                .border(1.dp, CyberTheme.PanelSlate)
                .padding(8.dp),
            contentAlignment = Alignment.Center
        ) {
            Column {
                // Header titles Row
                Row(modifier = Modifier.fillMaxWidth()) {
                    Text("", modifier = Modifier.weight(1.2f))
                    layerNames.forEach { name ->
                        Text(
                            text = name.take(5),
                            color = Color.Gray,
                            fontSize = 8.sp,
                            fontFamily = FontFamily.Monospace,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
                
                // Matrix cells grid
                for (r in 0 until 6) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 1.5.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Left title column
                        Text(
                            text = layerNames[r],
                            color = Color.LightGray,
                            fontSize = 9.sp,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.weight(1.2f)
                        )
                        
                        for (c in 0 until 6) {
                            val similarity = matrix[r][c]
                            val isSelf = r == c
                            
                            val squareBg = animateColorAsState(targetValue = getEmeraldThermalColor(similarity, isSelf)).value
                            
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .aspectRatio(1f)
                                    .padding(1.5.dp)
                                    .clip(RoundedCornerShape(3.dp))
                                    .background(squareBg),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = String.format("%.1f", similarity),
                                    color = if (similarity > 0.6f) Color.Black else Color.White,
                                    fontSize = 8.sp,
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------
// NEON GRADIENT RENDERING UTILS
// ----------------------------------------------------

fun getNeonThermalColor(score: Float): Color {
    val safeScore = score.coerceIn(0f, 1f)
    return when {
        safeScore < 0.2f -> Color(0xFF0D0E15) // Deep Slate
        safeScore < 0.5f -> Color(0xFF0D253F) // Dark Cyan
        safeScore < 0.8f -> CyberTheme.CnnCyan.copy(alpha = 0.85f) // Glowing Cyan
        else -> Color(0xFFFF1744) // Hot Crimson Core
    }
}

fun getAmberThermalColor(score: Float): Color {
    val safeScore = score.coerceIn(0f, 1f)
    return when {
        safeScore < 0.1f -> Color(0xFF0F1016)
        safeScore < 0.4f -> Color(0xFF3E2723) // Coffee Dark Amber
        safeScore < 0.8f -> CyberTheme.DtAmber.copy(alpha = 0.8f) // Amber
        else -> Color(0xFFFFCC00) // Highlight Yellow core
    }
}

fun getEmeraldThermalColor(similarity: Float, isSelf: Boolean): Color {
    val safeSim = similarity.coerceIn(0f, 1f)
    if (isSelf) return CyberTheme.DqnEmerald // Self identity is always bright green
    
    return when {
        safeSim < 0.2f -> Color(0xFF0F1016)
        safeSim < 0.5f -> Color(0xFF1B5E20) // Deep Forest Green
        safeSim < 0.8f -> CyberTheme.DqnEmerald.copy(alpha = 0.7f) // Emerald Green
        else -> Color(0xFFB9F6CA) // Light Mint White Green
    }
}
