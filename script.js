// --- Neural Network Logic ---

class NeuralNetwork {
    constructor(numInputs, numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        // Weights matrix: [output_index][input_index]
        // e.g., weights[0] is an array of 16 weights connecting to Output 0
        this.weights = [];
        this.biases = [];
        this.learningRate = 0.5;

        // Initialize with random weights between -1 and 1
        for (let i = 0; i < numOutputs; i++) {
            let row = [];
            for (let j = 0; j < numInputs; j++) {
                row.push(Math.random() * 2 - 1);
            }
            this.weights.push(row);
            this.biases.push(Math.random() * 2 - 1);
        }

        // Store last forward pass data for backprop
        this.lastInputs = [];
        this.lastOutputs = [];
    }

    // Sigmoid activation function
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid
    sigmoidDerivative(x) {
        return x * (1 - x); // Assuming x is already passed through sigmoid
    }

    forwardPass(inputs) {
        this.lastInputs = [...inputs];
        let outputs = [];

        for (let i = 0; i < this.numOutputs; i++) {
            let sum = this.biases[i];
            for (let j = 0; j < this.numInputs; j++) {
                sum += inputs[j] * this.weights[i][j];
            }
            outputs.push(this.sigmoid(sum));
        }

        this.lastOutputs = [...outputs];
        return outputs;
    }

    // Returns the calculated errors and weight updates (gradients) without applying them immediately
    // so we can animate them
    calculateBackpropUpdates(targets) {
        let errors = [];
        let deltas = [];
        let weightUpdates = []; // [output_index][input_index]

        // Calculate errors and deltas for outputs
        for (let i = 0; i < this.numOutputs; i++) {
            let error = targets[i] - this.lastOutputs[i];
            errors.push(error);
            let delta = error * this.sigmoidDerivative(this.lastOutputs[i]);
            deltas.push(delta);
        }

        // Calculate weight updates
        for (let i = 0; i < this.numOutputs; i++) {
            let rowUpdates = [];
            for (let j = 0; j < this.numInputs; j++) {
                // Update = learning_rate * delta * input
                let update = this.learningRate * deltas[i] * this.lastInputs[j];
                rowUpdates.push(update);
            }
            weightUpdates.push(rowUpdates);
        }

        return { errors, weightUpdates };
    }

    applyUpdates(weightUpdates) {
        for (let i = 0; i < this.numOutputs; i++) {
            for (let j = 0; j < this.numInputs; j++) {
                this.weights[i][j] += weightUpdates[i][j];
                // Clamp weights for visual stability between -2 and 2
                this.weights[i][j] = Math.max(-2, Math.min(2, this.weights[i][j]));
            }
        }
    }
}

// Ensure NN logic exists before UI code runs.
window.NeuralNetwork = NeuralNetwork;

// --- UI and Animation Logic ---

document.addEventListener('DOMContentLoaded', () => {
    const NUM_INPUTS = 16;
    const NUM_OUTPUTS = 4;

    // Initialize network
    let nn = new NeuralNetwork(NUM_INPUTS, NUM_OUTPUTS);

    // DOM Elements
    const inputMatrix = document.getElementById('input-matrix');
    const slidersContainer = document.getElementById('sliders-container');
    const outputNodesContainer = document.getElementById('output-nodes');

    const svgConnections = document.getElementById('connections-svg');
    const svgConnectionsOut = document.getElementById('connections-svg-out');

    const btnForward = document.getElementById('btn-forward');
    const btnBackprop = document.getElementById('btn-backprop');
    const btnReset = document.getElementById('btn-reset');
    const statusText = document.getElementById('status-text');

    const targetValuesContainer = document.getElementById('target-values');
    const errorValuesContainer = document.getElementById('error-values');

    // State
    let currentInputs = [];
    let currentTargets = [0.9, 0.1, 0.8, 0.2]; // Predefined targets for demonstration

    // --- 1. Setup UI Elements ---

    function setupUI() {
        // Clear containers
        inputMatrix.innerHTML = '';
        slidersContainer.innerHTML = '';
        outputNodesContainer.innerHTML = '';
        svgConnections.innerHTML = '';
        svgConnectionsOut.innerHTML = '';
        targetValuesContainer.innerHTML = '';
        errorValuesContainer.innerHTML = '';

        // Generate 16 Inputs
        for (let i = 0; i < NUM_INPUTS; i++) {
            let node = document.createElement('div');
            node.className = 'input-node';
            node.id = `input-${i}`;
            // Random binary input for demo
            let val = Math.random() > 0.5 ? 1 : 0;
            currentInputs.push(val);
            node.innerText = val;
            if (val === 1) node.classList.add('active');
            inputMatrix.appendChild(node);
        }

        // Generate 4 Rows of 16 Sliders (Weights)
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            let row = document.createElement('div');
            row.className = 'slider-row';
            row.dataset.label = `Out ${i+1}`;
            row.id = `slider-row-${i}`;

            for (let j = 0; j < NUM_INPUTS; j++) {
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

                track.appendChild(fill);
                track.appendChild(thumb);

                let valText = document.createElement('div');
                valText.className = 'slider-value-text';
                valText.id = `slider-val-${i}-${j}`;

                wrapper.appendChild(track);
                wrapper.appendChild(valText);
                row.appendChild(wrapper);
            }
            slidersContainer.appendChild(row);
        }

        // Generate 4 Outputs
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            let node = document.createElement('div');
            node.className = 'output-node';
            node.id = `output-${i}`;
            node.innerText = '0.00';
            node.dataset.targetLabel = `Target: ${currentTargets[i]}`;
            outputNodesContainer.appendChild(node);

            // Add to info panel
            let targetDiv = document.createElement('div');
            targetDiv.className = 'info-val';
            targetDiv.innerText = `O${i+1}: ${currentTargets[i]}`;
            targetValuesContainer.appendChild(targetDiv);

            let errorDiv = document.createElement('div');
            errorDiv.className = 'info-val';
            errorDiv.id = `error-val-${i}`;
            errorDiv.innerText = `O${i+1}: -`;
            errorValuesContainer.appendChild(errorDiv);
        }

        updateSlidersUI();

        // Wait for layout to draw lines
        setTimeout(() => {
            drawConnections();
        }, 100);
    }

    // Update Slider UI based on NN weights
    function updateSlidersUI() {
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            for (let j = 0; j < NUM_INPUTS; j++) {
                let weight = nn.weights[i][j];
                // Map weight [-2, 2] to percentage [0%, 100%]
                let percentage = ((weight + 2) / 4) * 100;
                // Clamp
                percentage = Math.max(0, Math.min(100, percentage));

                document.getElementById(`slider-fill-${i}-${j}`).style.height = `${percentage}%`;
                document.getElementById(`slider-thumb-${i}-${j}`).style.bottom = `${percentage}%`;
                document.getElementById(`slider-val-${i}-${j}`).innerText = weight.toFixed(2);
            }
        }
    }

    // --- 2. Drawing Connections (Lines) ---

    function drawConnections() {
        svgConnections.innerHTML = '';
        svgConnectionsOut.innerHTML = '';

        const networkRect = document.getElementById('network-container').getBoundingClientRect();

        // Connect Inputs to Sliders
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            for (let j = 0; j < NUM_INPUTS; j++) {
                let inputEl = document.getElementById(`input-${j}`);
                let sliderEl = document.getElementById(`slider-thumb-${i}-${j}`);

                let inRect = inputEl.getBoundingClientRect();
                let slRect = sliderEl.getBoundingClientRect();

                // Calculate relative positions
                let startX = inRect.right - networkRect.left;
                let startY = inRect.top + inRect.height / 2 - networkRect.top;

                let endX = slRect.left - networkRect.left;
                let endY = slRect.top + slRect.height / 2 - networkRect.top;

                let line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                let d = `M ${startX} ${startY} C ${startX + 50} ${startY}, ${endX - 50} ${endY}, ${endX} ${endY}`;

                line.setAttribute('d', d);
                line.setAttribute('class', 'connection-line');
                line.id = `line-in-${i}-${j}`;

                // Initial line width based on absolute weight
                let weight = Math.abs(nn.weights[i][j]);
                line.style.strokeWidth = Math.max(0.5, weight * 2) + 'px';

                svgConnections.appendChild(line);
            }
        }

        // Connect Sliders to Outputs
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            let outEl = document.getElementById(`output-${i}`);
            let outRect = outEl.getBoundingClientRect();
            let endX = outRect.left - networkRect.left;
            let endY = outRect.top + outRect.height / 2 - networkRect.top;

            // Connect from the end of the slider row to the output node
            let rowEl = document.getElementById(`slider-row-${i}`);
            let rowRect = rowEl.getBoundingClientRect();
            let startX = rowRect.right - networkRect.left;
            let startY = rowRect.top + rowRect.height / 2 - networkRect.top;

            let line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            let d = `M ${startX} ${startY} C ${startX + 50} ${startY}, ${endX - 50} ${endY}, ${endX} ${endY}`;

            line.setAttribute('d', d);
            line.setAttribute('class', 'connection-line');
            line.id = `line-out-${i}`;
            line.style.strokeWidth = '2px';

            svgConnectionsOut.appendChild(line);
        }
    }

    // Resize observer to redraw lines
    window.addEventListener('resize', () => {
        drawConnections();
    });

    // --- 3. Animation & Logic ---

    async function runForwardPass() {
        btnForward.disabled = true;
        btnReset.disabled = true;
        statusText.innerText = "Running Forward Pass...";

        // Light up input lines
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            for (let j = 0; j < NUM_INPUTS; j++) {
                if (currentInputs[j] === 1) {
                    let line = document.getElementById(`line-in-${i}-${j}`);
                    line.classList.add('forward');
                }
            }
        }

        await new Promise(r => setTimeout(r, 800)); // Delay for visual

        // Calculate outputs
        let outputs = nn.forwardPass(currentInputs);

        // Light up output lines
        for (let i = 0; i < NUM_OUTPUTS; i++) {
             document.getElementById(`line-out-${i}`).classList.add('forward');
             let outNode = document.getElementById(`output-${i}`);
             outNode.innerText = outputs[i].toFixed(2);
             outNode.classList.add('active');
        }

        await new Promise(r => setTimeout(r, 500));

        statusText.innerText = "Forward Pass Complete. Ready for Backprop.";
        btnBackprop.disabled = false;
        btnReset.disabled = false;
    }

    async function runBackprop() {
        btnForward.disabled = true;
        btnBackprop.disabled = true;
        btnReset.disabled = true;
        statusText.innerText = "Calculating Error...";

        let { errors, weightUpdates } = nn.calculateBackpropUpdates(currentTargets);

        // 1. Show Errors at outputs
        for (let i = 0; i < NUM_OUTPUTS; i++) {
             let outNode = document.getElementById(`output-${i}`);
             outNode.style.backgroundColor = '#f44336'; // Red for error
             outNode.style.borderColor = '#ffcdd2';
             outNode.innerText = `Err: ${errors[i].toFixed(2)}`;

             document.getElementById(`error-val-${i}`).innerText = `O${i+1}: ${errors[i].toFixed(4)}`;
        }

        await new Promise(r => setTimeout(r, 1000));

        statusText.innerText = "Backpropagating Error...";

        // 2. Animate backward lines
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            document.getElementById(`line-out-${i}`).classList.remove('forward');
            document.getElementById(`line-out-${i}`).classList.add('backward');

            for (let j = 0; j < NUM_INPUTS; j++) {
                let line = document.getElementById(`line-in-${i}-${j}`);
                line.classList.remove('forward');
                // Only show significant backflow
                if (Math.abs(weightUpdates[i][j]) > 0.01) {
                    line.classList.add('backward');
                }
            }
        }

        await new Promise(r => setTimeout(r, 1500));

        statusText.innerText = "Updating Weights (Sliders)...";

        // 3. Apply updates to NN logic
        nn.applyUpdates(weightUpdates);

        // 4. Update Sliders UI and Line Widths
        updateSlidersUI();

        for (let i = 0; i < NUM_OUTPUTS; i++) {
            for (let j = 0; j < NUM_INPUTS; j++) {
                let line = document.getElementById(`line-in-${i}-${j}`);
                // Update line thickness based on new weight
                let newWeight = Math.abs(nn.weights[i][j]);
                line.style.strokeWidth = Math.max(0.5, newWeight * 2) + 'px';
            }
        }

        await new Promise(r => setTimeout(r, 1000));

        // Cleanup animations
        for (let i = 0; i < NUM_OUTPUTS; i++) {
            document.getElementById(`line-out-${i}`).classList.remove('backward');
            let outNode = document.getElementById(`output-${i}`);
            outNode.style.backgroundColor = '';
            outNode.style.borderColor = '';
            outNode.classList.remove('active');
            outNode.innerText = '0.00';

            for (let j = 0; j < NUM_INPUTS; j++) {
                document.getElementById(`line-in-${i}-${j}`).classList.remove('backward');
            }
        }

        statusText.innerText = "Training Step Complete. Ready for next Forward Pass.";
        btnForward.disabled = false;
        btnReset.disabled = false;
    }

    // --- 4. Event Listeners ---

    btnForward.addEventListener('click', runForwardPass);
    btnBackprop.addEventListener('click', runBackprop);

    btnReset.addEventListener('click', () => {
        nn = new NeuralNetwork(NUM_INPUTS, NUM_OUTPUTS);
        currentInputs = [];
        btnBackprop.disabled = true;
        btnForward.disabled = false;
        statusText.innerText = "Network Reset. Click 'Forward Pass' to start.";
        setupUI();
    });

    // Initialize
    setupUI();
});
