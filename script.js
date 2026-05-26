// --- Neural Network Logic ---

class NeuralNetwork {
    constructor(numInputs, numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        // Weights matrix: [input_index][output_index]
        // This makes it easier to map to 16 rows x 4 cols UI
        this.weights = [];
        this.biases = [];
        this.learningRate = 0.05;

        // Initialize empty, user will trigger init
        for (let i = 0; i < numInputs; i++) {
            this.weights.push(new Array(numOutputs).fill(0));
        }
        this.biases = new Array(numOutputs).fill(0);

        this.lastInputs = [];
        this.lastOutputs = [];
        this.lossHistory = [];
    }

    initializeRandom() {
        for (let i = 0; i < this.numInputs; i++) {
            for (let j = 0; j < this.numOutputs; j++) {
                // Initialize between -1 and 1
                this.weights[i][j] = Math.random() * 2 - 1;
            }
        }
        for (let j = 0; j < this.numOutputs; j++) {
            this.biases[j] = Math.random() * 2 - 1;
        }
    }

    // ReLU activation
    relu(x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU
    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    forwardPass(inputs) {
        this.lastInputs = [...inputs];
        let outputs = [];

        for (let j = 0; j < this.numOutputs; j++) {
            let sum = this.biases[j];
            for (let i = 0; i < this.numInputs; i++) {
                sum += inputs[i] * this.weights[i][j];
            }
            // Apply ReLU
            outputs.push(this.relu(sum));
        }

        this.lastOutputs = [...outputs];
        return outputs;
    }

    calculateBackpropUpdates(targets) {
        let errors = [];
        let deltas = [];
        let weightUpdates = []; // [input_index][output_index]
        let loss = 0;

        // Calculate errors, deltas, and MSE loss
        for (let j = 0; j < this.numOutputs; j++) {
            let error = targets[j] - this.lastOutputs[j];
            errors.push(error);
            loss += error * error;

            // Note: error is (target - output), we want to maximize it (gradient descent subtracts grad)
            // Or if standard grad descent: Delta = -error * deriv
            // To keep update = learningRate * delta * input, we use delta = error * deriv
            let delta = error * this.reluDerivative(this.lastOutputs[j]);
            deltas.push(delta);
        }

        // MSE
        loss = loss / this.numOutputs;
        this.lossHistory.push(loss);

        // Initialize weight updates array structure
        for (let i = 0; i < this.numInputs; i++) {
            weightUpdates.push(new Array(this.numOutputs).fill(0));
        }

        // Calculate weight updates
        for (let i = 0; i < this.numInputs; i++) {
            for (let j = 0; j < this.numOutputs; j++) {
                weightUpdates[i][j] = this.learningRate * deltas[j] * this.lastInputs[i];
            }
        }

        return { errors, weightUpdates, loss };
    }

    applyUpdates(weightUpdates) {
        for (let i = 0; i < this.numInputs; i++) {
            for (let j = 0; j < this.numOutputs; j++) {
                this.weights[i][j] += weightUpdates[i][j];
                // Clamp weights for UI display [-2, 2]
                this.weights[i][j] = Math.max(-2, Math.min(2, this.weights[i][j]));
            }
        }
    }
}

window.NeuralNetwork = NeuralNetwork;

// --- UI and Animation Logic ---

document.addEventListener('DOMContentLoaded', () => {
    const NUM_INPUTS = 16;
    const NUM_OUTPUTS = 4;

    // Initialize network
    let nn = new NeuralNetwork(NUM_INPUTS, NUM_OUTPUTS);

    // DOM Elements
    const inputQueueContainer = document.getElementById('input-queue');
    const slidersContainer = document.getElementById('sliders-container');
    const outputNodesContainer = document.getElementById('output-nodes');

    const svgConnections = document.getElementById('connections-svg');
    const svgConnectionsOut = document.getElementById('connections-svg-out');

    const btnInit = document.getElementById('btn-init');
    const btnTrain = document.getElementById('btn-train-step');
    const btnAutoPlay = document.getElementById('btn-autoplay');
    const delayInput = document.getElementById('delay-input');
    const btnReset = document.getElementById('btn-reset');
    const statusText = document.getElementById('status-text');

    let lossChart;

    // State
    let inputHistory = []; // Array of arrays (16 values each)
    let currentInputs = [];
    let currentTargets = [];
    let isInitialized = false;
    let stepCount = 0;
    let isAutoPlaying = false;

    // --- 1. Setup UI Elements ---

    function setupUI() {
        // Clear containers
        inputQueueContainer.innerHTML = '';
        slidersContainer.innerHTML = '';
        outputNodesContainer.innerHTML = '';
        svgConnections.innerHTML = '';
        svgConnectionsOut.innerHTML = '';

        // Generate Input Queue (4 columns: history-3, history-2, history-1, current)
        for (let col = 3; col >= 0; col--) {
            let colDiv = document.createElement('div');
            colDiv.className = `input-column ${col === 0 ? 'current' : `history-${col}`}`;
            colDiv.id = `input-col-${col}`;

            for (let i = 0; i < NUM_INPUTS; i++) {
                let node = document.createElement('div');
                node.className = 'input-node';
                node.id = `input-${col}-${i}`;
                node.innerText = '0';
                colDiv.appendChild(node);
            }
            inputQueueContainer.appendChild(colDiv);
        }

        // Generate Weights (16 rows, 4 horizontal sliders each)
        for (let i = 0; i < NUM_INPUTS; i++) {
            let row = document.createElement('div');
            row.className = 'slider-row';
            row.dataset.inLabel = `I${i}`;
            row.id = `slider-row-${i}`;

            for (let j = 0; j < NUM_OUTPUTS; j++) {
                let wrapper = document.createElement('div');
                wrapper.className = 'slider-wrapper';

                let track = document.createElement('div');
                track.className = 'slider-track';

                let fill = document.createElement('div');
                fill.className = 'slider-fill';
                fill.id = `slider-fill-${i}-${j}`;

                let thumb = document.createElement('div');
                thumb.className = 'slider-thumb';
                thumb.id = `slider-thumb-${i}-${j}`;

                let valText = document.createElement('div');
                valText.className = 'slider-value-text';
                valText.id = `slider-val-${i}-${j}`;
                valText.textContent = "0.00";

                track.appendChild(fill);
                track.appendChild(thumb);
                wrapper.appendChild(track);
                wrapper.appendChild(valText);
                row.appendChild(wrapper);
            }
            slidersContainer.appendChild(row);
        }

        // Generate Outputs (ReLU Dials + Target Tracks)
        for (let j = 0; j < NUM_OUTPUTS; j++) {
            let outContainer = document.createElement('div');
            outContainer.className = 'output-container';
            outContainer.id = `output-container-${j}`;

            // ReLU Dial
            let dialContainer = document.createElement('div');
            dialContainer.className = 'dial-container';
            dialContainer.id = `dial-container-${j}`;

            let dialBg = document.createElement('div');
            dialBg.className = 'dial-bg';
            dialBg.id = `dial-bg-${j}`;

            let dialCenter = document.createElement('div');
            dialCenter.className = 'dial-center';
            dialCenter.id = `dial-val-${j}`;
            dialCenter.innerText = '0.00';

            dialContainer.appendChild(dialBg);
            dialContainer.appendChild(dialCenter);

            // Target Track
            let targetTrack = document.createElement('div');
            targetTrack.className = 'target-track';
            targetTrack.id = `target-track-${j}`;

            let label = document.createElement('div');
            label.className = 'target-label';
            label.innerText = `Out ${j}`;

            targetTrack.appendChild(label);

            // Error Text (hidden initially)
            let errorText = document.createElement('div');
            errorText.className = 'error-text';
            errorText.id = `error-text-${j}`;

            outContainer.appendChild(dialContainer);
            outContainer.appendChild(targetTrack);
            outContainer.appendChild(errorText);

            outputNodesContainer.appendChild(outContainer);
        }

        setupChart();
        updateSlidersUI();

        setTimeout(() => {
            drawConnections();
        }, 100);
    }

    function setupChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        if (lossChart) lossChart.destroy();
        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'MSE Loss',
                    data: [],
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#aaa' },
                        grid: { color: '#333' }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#aaa' },
                        grid: { color: '#333' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function generateNewDataPair() {
        // Generate random inputs [0 or 1]
        let inputs = [];
        for (let i = 0; i < NUM_INPUTS; i++) {
            inputs.push(Math.random() > 0.5 ? 1 : 0);
        }

        // Target: simple dummy logic (e.g., target 1 sum of first 4 inputs, normalized)
        let targets = [];
        for (let j = 0; j < NUM_OUTPUTS; j++) {
            // Random target between 0 and 2 for ReLU demonstration
            targets.push(Math.random() * 2);
        }
        return { inputs, targets };
    }

    // --- 2. Drawing Connections (Lines) ---

    function drawConnections() {
        svgConnections.innerHTML = '';
        svgConnectionsOut.innerHTML = '';

        const networkRect = document.getElementById('network-container').getBoundingClientRect();

        // Connect Current Input (col 0) to Weight Rows
        for (let i = 0; i < NUM_INPUTS; i++) {
            let inputEl = document.getElementById(`input-0-${i}`);
            let rowEl = document.getElementById(`slider-row-${i}`);

            let inRect = inputEl.getBoundingClientRect();
            let rowRect = rowEl.getBoundingClientRect();

            let startX = inRect.right - networkRect.left;
            let startY = inRect.top + inRect.height / 2 - networkRect.top;

            let endX = rowRect.left - networkRect.left;
            let endY = rowRect.top + rowRect.height / 2 - networkRect.top;

            let line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            let d = `M ${startX} ${startY} C ${startX + 30} ${startY}, ${endX - 30} ${endY}, ${endX} ${endY}`;

            line.setAttribute('d', d);
            line.setAttribute('class', 'connection-line');
            line.id = `line-in-${i}`;
            line.style.strokeWidth = '1px';

            svgConnections.appendChild(line);
        }

        // Connect Weight Rows to Outputs
        // We will draw 4 lines from each row, targeting the 4 output dials
        for (let i = 0; i < NUM_INPUTS; i++) {
            for (let j = 0; j < NUM_OUTPUTS; j++) {
                let rowEl = document.getElementById(`slider-row-${i}`);
                let dialEl = document.getElementById(`dial-container-${j}`);

                let rowRect = rowEl.getBoundingClientRect();
                let dialRect = dialEl.getBoundingClientRect();

                // Estimate position of the specific slider in the row (roughly dividing row into 4)
                let sliderWidth = rowRect.width / 4;
                let startX = rowRect.left + (sliderWidth * j) + (sliderWidth / 2) - networkRect.left;
                let startY = rowRect.bottom - networkRect.top; // Bottom of row

                let endX = dialRect.left - networkRect.left;
                let endY = dialRect.top + dialRect.height / 2 - networkRect.top;

                let line = document.createElementNS('http://www.w3.org/2000/svg', 'path');

                // Using a curve that comes from bottom of slider to left of dial
                let d = `M ${startX} ${startY} C ${startX} ${startY + 50}, ${endX - 50} ${endY}, ${endX} ${endY}`;

                line.setAttribute('d', d);
                line.setAttribute('class', 'connection-line');
                line.id = `line-out-${i}-${j}`;

                let weight = Math.abs(nn.weights[i][j]);
                line.style.strokeWidth = Math.max(0.5, weight * 2) + 'px';

                svgConnectionsOut.appendChild(line);
            }
        }
    }

    window.addEventListener('resize', drawConnections);

    // --- 3. UI Updates ---

    function updateSlidersUI() {
        for (let i = 0; i < NUM_INPUTS; i++) {
            for (let j = 0; j < NUM_OUTPUTS; j++) {
                let weight = nn.weights[i][j];
                // Map [-2, 2] to [0%, 100%]
                let pct = ((weight + 2) / 4) * 100;
                pct = Math.max(0, Math.min(100, pct));

                document.getElementById(`slider-fill-${i}-${j}`).style.width = `${pct}%`;
                document.getElementById(`slider-thumb-${i}-${j}`).style.left = `${pct}%`;

                let valText = document.getElementById(`slider-val-${i}-${j}`);
                valText.innerText = weight.toFixed(2);

                if (weight >= 0) {
                     valText.style.left = '5px';
                     valText.style.right = 'auto';
                } else {
                     valText.style.right = '5px';
                     valText.style.left = 'auto';
                }
                let line = document.getElementById(`line-out-${i}-${j}`);
                if (line) {
                    line.style.strokeWidth = Math.max(0.5, Math.abs(weight) * 2) + 'px';
                }
            }
        }
    }

    function updateInputQueueUI() {
        for (let col = 3; col >= 0; col--) {
            let data = col === 0 ? currentInputs : inputHistory[col - 1];
            if (!data) continue;

            for (let i = 0; i < NUM_INPUTS; i++) {
                let node = document.getElementById(`input-${col}-${i}`);
                let val = data[i];
                node.innerText = val;
                if (val === 1) {
                    node.classList.add('active');
                } else {
                    node.classList.remove('active');
                }
            }
        }
    }

    function setOutputDial(index, value) {
        let dialBg = document.getElementById(`dial-bg-${index}`);
        let dialVal = document.getElementById(`dial-val-${index}`);

        dialVal.innerText = value.toFixed(2);

        // Map ReLU output [0, ~2] to a degree [0, 360] for visual
        let degrees = Math.min(360, (value / 2) * 360);
        dialBg.style.background = `conic-gradient(#9C27B0 ${degrees}deg, #555 ${degrees}deg)`;

        if (value > 0) {
            dialBg.classList.add('active');
        } else {
            dialBg.classList.remove('active');
        }
    }

    function updateChart() {
        lossChart.data.labels.push(stepCount);
        lossChart.data.datasets[0].data.push(nn.lossHistory[nn.lossHistory.length - 1]);
        lossChart.update();
    }

    // --- 4. Animations ---

    async function initializeNetwork() {
        btnInit.disabled = true;
        btnTrain.disabled = true;
        btnReset.disabled = true;
        statusText.innerText = "Initializing random weights...";

        // Add scrambling class to UI
        slidersContainer.classList.add('scrambling');
        outputNodesContainer.classList.add('scrambling');

        // Visually rapid update values
        for (let t = 0; t < 20; t++) {
            for (let i = 0; i < NUM_INPUTS; i++) {
                for (let j = 0; j < NUM_OUTPUTS; j++) {
                    let rPct = Math.random() * 100;
                    document.getElementById(`slider-fill-${i}-${j}`).style.width = `${rPct}%`;
                    document.getElementById(`slider-thumb-${i}-${j}`).style.left = `${rPct}%`;
                }
            }
            await new Promise(r => setTimeout(r, 50));
        }

        slidersContainer.classList.remove('scrambling');
        outputNodesContainer.classList.remove('scrambling');

        // Actually initialize the logic
        nn.initializeRandom();
        updateSlidersUI();

        // Initialize history
        inputHistory = [new Array(16).fill(0), new Array(16).fill(0), new Array(16).fill(0)];
        let initData = generateNewDataPair();
        currentInputs = initData.inputs;
        currentTargets = initData.targets;

        updateInputQueueUI();

        isInitialized = true;
        statusText.innerText = "Initialization complete. Ready to train.";
        btnTrain.disabled = false;
        btnAutoPlay.disabled = false;
        btnReset.disabled = false;
    }

    function toggleAutoPlay() {
        if (!isInitialized) return;
        isAutoPlaying = !isAutoPlaying;

        if (isAutoPlaying) {
            btnAutoPlay.innerText = "Stop Auto Play";
            btnAutoPlay.classList.add('active-play');
            btnInit.disabled = true;
            btnTrain.disabled = true;
            btnReset.disabled = true;
            delayInput.disabled = true;
            runTrainStep();
        } else {
            btnAutoPlay.innerText = "Auto Play";
            btnAutoPlay.classList.remove('active-play');
            btnInit.disabled = false;
            btnTrain.disabled = false;
            btnReset.disabled = false;
            delayInput.disabled = false;
            statusText.innerText = `Auto play stopped. Step ${stepCount} Complete.`;
        }
    }

    async function runTrainStep() {
        if (!isInitialized) return;
        btnTrain.disabled = true;
        btnInit.disabled = true;
        btnReset.disabled = true;
        stepCount++;

        // 1. Shift Queue & Slide Targets
        statusText.innerText = "Step " + stepCount + ": Loading inputs & targets...";
        inputHistory.unshift([...currentInputs]);
        if (inputHistory.length > 3) inputHistory.pop();

        let newData = generateNewDataPair();
        currentInputs = newData.inputs;
        let newTargets = newData.targets;

        updateInputQueueUI();

        // Animate Targets
        let targetElements = [];
        for (let j = 0; j < NUM_OUTPUTS; j++) {
            let track = document.getElementById(`target-track-${j}`);

            // Create new incoming target
            let newTargetEl = document.createElement('div');
            newTargetEl.className = 'target-item incoming';
            newTargetEl.innerText = newTargets[j].toFixed(2);
            track.appendChild(newTargetEl);
            targetElements.push(newTargetEl);

            // Move existing to outgoing
            let existing = track.querySelectorAll('.target-item.active');
            existing.forEach(el => {
                el.classList.remove('active');
                el.classList.add('outgoing');
                setTimeout(() => el.remove(), 1000);
            });
        }

        // Trigger CSS transition
        await new Promise(r => setTimeout(r, 50));
        targetElements.forEach(el => {
            el.classList.remove('incoming');
            el.classList.add('active');
        });

        currentTargets = newTargets;
        await new Promise(r => setTimeout(r, 1000));

        // 2. Forward Pass
        statusText.innerText = "Forward Pass (ReLU)...";
        for (let i = 0; i < NUM_INPUTS; i++) {
            if (currentInputs[i] === 1) {
                document.getElementById(`line-in-${i}`).classList.add('forward');
            }
        }

        await new Promise(r => setTimeout(r, 500));

        let outputs = nn.forwardPass(currentInputs);

        for (let j = 0; j < NUM_OUTPUTS; j++) {
            setOutputDial(j, outputs[j]);
            for (let i = 0; i < NUM_INPUTS; i++) {
                if (currentInputs[i] === 1) {
                   document.getElementById(`line-out-${i}-${j}`).classList.add('forward');
                }
            }
        }

        await new Promise(r => setTimeout(r, 1000));

        // 3. Backpropagate
        statusText.innerText = "Calculating Error & Backpropagating...";
        let { errors, weightUpdates, loss } = nn.calculateBackpropUpdates(currentTargets);

        // Show errors
        for (let j = 0; j < NUM_OUTPUTS; j++) {
            let errText = document.getElementById(`error-text-${j}`);
            errText.innerText = `E: ${errors[j].toFixed(2)}`;
            errText.classList.add('show');
        }

        await new Promise(r => setTimeout(r, 800));

        // Backward flow visual
        for (let i = 0; i < NUM_INPUTS; i++) {
            document.getElementById(`line-in-${i}`).classList.remove('forward');
            for (let j = 0; j < NUM_OUTPUTS; j++) {
                document.getElementById(`line-out-${i}-${j}`).classList.remove('forward');
                if (Math.abs(weightUpdates[i][j]) > 0.005) {
                    document.getElementById(`line-out-${i}-${j}`).classList.add('backward');
                }
            }
        }

        await new Promise(r => setTimeout(r, 1000));
        statusText.innerText = "Updating Weights...";

        nn.applyUpdates(weightUpdates);
        updateSlidersUI();
        updateChart();

        await new Promise(r => setTimeout(r, 1000));

        // Cleanup
        for (let i = 0; i < NUM_INPUTS; i++) {
            for (let j = 0; j < NUM_OUTPUTS; j++) {
                document.getElementById(`line-out-${i}-${j}`).classList.remove('backward');
            }
        }
        for (let j = 0; j < NUM_OUTPUTS; j++) {
            document.getElementById(`error-text-${j}`).classList.remove('show');
            document.getElementById(`dial-bg-${j}`).style.background = 'conic-gradient(#555 0%, #333 0%)';
            document.getElementById(`dial-bg-${j}`).classList.remove('active');
            document.getElementById(`dial-val-${j}`).innerText = '0.00';
        }

        if (isAutoPlaying) {
            let delayMs = parseInt(delayInput.value) || 0;
            statusText.innerText = `Step ${stepCount} Complete. Waiting ${delayMs}ms before next step...`;
            await new Promise(r => setTimeout(r, delayMs));

            // Re-check after delay in case user clicked stop during wait
            if (isAutoPlaying) {
                runTrainStep();
            }
        } else {
            statusText.innerText = `Step ${stepCount} Complete. Ready for next step.`;
            btnInit.disabled = false;
            btnTrain.disabled = false;
            btnReset.disabled = false;
        }
    }

    // --- 5. Event Listeners ---

    btnInit.addEventListener('click', initializeNetwork);
    btnTrain.addEventListener('click', runTrainStep);
    btnAutoPlay.addEventListener('click', toggleAutoPlay);

    btnReset.addEventListener('click', () => {
        nn = new NeuralNetwork(NUM_INPUTS, NUM_OUTPUTS);
        inputHistory = [];
        currentInputs = [];
        currentTargets = [];
        isInitialized = false;
        stepCount = 0;
        isAutoPlaying = false;

        btnInit.disabled = false;
        btnTrain.disabled = true;
        btnAutoPlay.disabled = true;
        btnAutoPlay.innerText = "Auto Play";
        btnAutoPlay.classList.remove('active-play');
        delayInput.disabled = false;

        statusText.innerText = "Network Reset. Click 'Initialize Weights' to start.";
        setupUI();
    });

    // Run startup layout
    setupUI();
});
