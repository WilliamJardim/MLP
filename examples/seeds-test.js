let model = net.MLP({
    initialization: 'zeros',

    hyperparameters: {
        learningRate: 0.5
    },

    //Structure
    layers: [
        { type: 'input',  inputs: 2, pre_process : 'nothing' }, //The input Layer
        { type: 'hidden', inputs: 2, units : 2, activation : 'sigmoid' },
        { type: 'output', inputs: 2, units : 2, activation : 'sigmoid' }
    ]
});

//Define the same weights used in Machine Learning Mastery
model.layers[0].units[0].weights = [0.13436424411240122, 0.8474337369372327];
model.layers[0].units[0].bias = 0.763774618976614;

model.layers[0].units[1].weights = [0.2550690257394217, 0.49543508709194095];
model.layers[0].units[1].bias = 0.4494910647887381;

model.layers[1].units[0].weights = [0.651592972722763, 0.7887233511355132];
model.layers[1].units[0].bias = 0.0938595867742349;

model.layers[1].units[1].weights = [0.02834747652200631, 0.8357651039198697, 0.43276706790505337];
model.layers[1].units[1].bias = 0.43276706790505337;

//XOR problem
//dataset is [ features, targets_values ]
let dataset = [
    [ [2.7810836, 2.550537003],     [1,0] ],
    [ [1.465489372, 2.362125076],   [1,0] ],
    [ [3.396561688, 4.400293529],   [1,0] ],
    [ [1.38807019, 1.850220317],    [1,0] ],
    [ [3.06407232, 3.005305973],    [1,0] ],
    [ [7.627531214, 2.759262235],   [0,1] ],
    [ [5.332441248, 2.088626775],   [0,1] ],
    [ [6.922596716, 1.77106367],    [0,1] ], 
    [ [8.675418651, -0.242068655],  [0,1] ], 
    [ [7.673756466, 3.508563011],   [0,1] ]
]

//Training the model
let results = model.train(dataset, 20);

//AS THE RESULT, THE FINAL WEIGHTS AFTER TRAINING IS CLOSER TO THE EXEMPLE OF MACHINE LEARNING MASTERY ARTICLE
//AND THE BEHAVIOR OF THE ERROR REDUCING GRADUALLY DURING THE TRAINING IS THE SAME TOO

/*
let test_dataset = [
 [ [2.7810836,2.550537003],     0 ],
 [ [1.465489372,2.362125076],   0 ],
 [ [3.396561688,4.400293529],   0 ],
 [ [1.38807019,1.850220317],    0 ],
 [ [3.06407232,3.005305973],    0 ],
 [ [7.627531214,2.759262235],   1 ],
 [ [5.332441248,2.088626775],   1 ],
 [ [6.922596716,1.77106367],    1 ],
 [ [8.675418651,-0.242068655],  1 ],
 [ [7.673756466,3.508563011],   1 ]
]
 */