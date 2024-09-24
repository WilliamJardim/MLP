let model = new net.MLP({
    initialization: 'zeros',
    task: 'classification',
    traintype: 'online',

    hyperparameters: {
        learningRate: 0.001
    },

    //Structure
    layers: [
        { type: 'input',  inputs: 2, pre_process : 'nothing' }, //The input Layer
        { type: 'hidden', inputs: 2, units : 3, activation : 'sigmoid' },
        { type: 'output', inputs: 3, units : 2, activation : 'sigmoid' }
    ]
});

