let model = new net.MLP({
    initialization: 'zeros',
    task: 'binary_classification',
    traintype: 'online',

    hyperparameters: {
        learningRate: 0.5
    },

    //Structure
    layers: [
        { type: 'input',  inputs: 2, pre_process : 'nothing' }, //The input Layer
        { type: 'hidden', inputs: 2, units : 3, activation : 'relu' },
        { type: 'final',  inputs: 3, units : 1, activation : 'sigmoid' }
    ]
});

//XOR problem
//dataset is [ features, targets_values ]
let dataset = net.data.Dataset([
    [ [0, 1], [ 1 ] ],
    [ [1, 0], [ 1 ] ],
    [ [0, 0], [ 0 ] ],
    [ [1, 1], [ 0 ] ],
]);

//Training the model
let results = model.train(dataset, 256);

console.log('Initial LOSS', results.initial_loss);
console.log('Final LOSS', results.final_loss);