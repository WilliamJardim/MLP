/**
* Multilayer Perceptron Neural Network (MLP)
* By William Alves Jardim
* 
* This implementation is entirely original, written from scratch in JavaScript.
* It was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
* For more details, see README.md.
*/
if( !window.net ){
    window.net = {};
    window.net.training = {};
}

//The entire model
net.MLP = class{
    constructor( config_dict={} ){
        let context = this;

        context.config_dict        = config_dict;
        context.inputs_config      = config_dict['inputs_config'] || {};
        context.layers_structure   = config_dict['layers'] || [];
        context.number_of_layers   = context.layers_structure.length;
        context.input_layer        = context.layers_structure[0]; //Get the input layer
        context.last_layer         = context.layers_structure[context.number_of_layers-1];

        context.task               = config_dict['task'];
        context.training_type      = config_dict['traintype'];
        context.hyperparameters    = config_dict['hyperparameters'];
        context.learning_rate      = context.hyperparameters['learningRate'];
        context.vincules           = {};

        //Class parameters and model Hyperparameters validations
        if( context.training_type == undefined || context.training_type == null ){
            throw Error(`context.training_type is not defined!`);
        }
        if(context.number_of_layers == 0){
            throw Error(`The model does not have any layers!`);
        }
        if( config_dict['layers'] == undefined || config_dict['layers'] == null ){
            throw Error(`The model must have the 'layers' property in the config_dict`);
        }
        if( config_dict['hyperparameters'] == undefined || config_dict['hyperparameters'] == null ){
            throw Error(`The model must have the 'hyperparameters' property in the config_dict`);
        }
        if( context.input_layer == undefined || context.input_layer['type'] != 'input' ){
            throw Error(`context.input_layer is undefined OR is not of type INPUT!. The first layer must be the INPUT LAYER`);
        }
        if( context.last_layer == undefined || context.last_layer['type'] != 'output' ){
            throw Error(`context.last_layer is undefined OR is not of type OUTPUT!. The last layer must be the OUTPUT LAYER`);
        }
        if( context.learning_rate == undefined || context.learning_rate == null ){
            throw Error(`hyperparameters.learning_rate is undefined!`);
        }
        if( context.learning_rate == Infinity ){
            throw Error(`hyperparameters.learning_rate is Infinity!`);
        }
        if( isNaN(context.learning_rate) == true ){
            throw Error(`hyperparameters.learning_rate is NaN!`);
        }

        //Task(model use type) validations in initialization
        if( context.task == undefined || context.task == null ){
            throw Error(`context.task is not defined!`);
        }
        switch(context.task){
            case 'regression':
            case 'linear_regression':
                if( context.last_layer.activation != 'relu' ){
                    throw Error(`In the linear regression, you cannot use ${context.last_layer.activation} as output activation function!`);
                }
                break;

            case 'classification':
            case 'logistic_regression':
                if( context.last_layer.activation != 'sigmoid' ){
                    throw Error(`In the classification, you cannot use ${context.last_layer.activation} as output activation function!`);
                }

                break;

            //Only two classes
            case 'binary_classification':
                if( context.last_layer.activation != 'sigmoid' ){
                    throw Error(`In the binary classification, you cannot use ${context.last_layer.activation} as output activation function!`);
                }

                if( context.last_layer.units > 1 ){
                    throw Error(`In the binary classification, the number of outputs must be only 1, but is ${context.last_layer.units}`);
                }
                break;

            default:
                throw Error(`Invalid task: context.task=${context.task} !`);
                break;
        }

        //The layers objects that will be created below
        context.layers  = [];

        //If true, not allow modifications in layers
        context.layers_locked = true;

        /**
        * Get the own context
        * @returns {Object} - the model it self
        */
        context.getSelfContext = function(){
            return context;
        }

        /**
        * Add a layer to context.layers 
        * 
        * @param {net.Layer} new_layer_object
        */
        context.addLayer = function( new_layer_object ){

            if( typeof new_layer_object == 'object' && 
                new_layer_object instanceof Object &&
                new_layer_object.objectName == 'Layer'
            ){
                if( !context.layers_locked )
                {
                    context.layers = [...context.layers, new_layer_object];

                }else{
                    throw Error('The layers are locked for new additions!');
                }

            }else{
                throw Error('The "new_layer_object" is not a object of type Layer.');
            }

        }

        let last_created = null;

        /* 
        * Initialize the network (ignoring the first layer that is the input layer)
        * Because, the layer of index 1 is the first layer, and in the context.layers_structure Array, the index 0 is the input layer
        * And then, the index 1 is the first hidden layer, and is the reason by starting at index 1 in the "for loop" below:
        */
        context.layers_locked = false;
        for( let i = 1 ; i < context.number_of_layers ; i++ )
        {
            let current_layer = net.Layer( context.layers_structure[i], ( layerItSelf ) => {

                const model_context = context.getSelfContext();
                const layer_context = layerItSelf.getSelfContext();

                /**
                * Do important vincules 
                */
                layer_context.atSelf()
                             /*Here I used "i-1" precisely because we are ignoring the input layer, as this for loop starts at layer 1 forward (precisely to ignore the input layer)*/
                             .vinculate('_internal_index', i-1);

                layer_context.atSelf()
                             .vinculate('_father',         context);
                
                /* Add the layer in the model */
                model_context.atSelf()
                             .addLayer( layer_context );

            });

            //Validations of layer creation
            if( last_created && current_layer.number_of_inputs != last_created.number_of_units ) {
                throw Error(`Initialization error: The layer ${i} have ${ current_layer.number_of_inputs } inputs. But, should be ${ last_created.number_of_units } inputs, because the previus layer( the layer ${i-1} ) have ${ last_created.number_of_units } output units.`);
            }
            last_created = current_layer;
        }

        //Lock the layers
        context.layers_locked = true;

        //Final validations after initialization
        if( context.input_layer.type != 'input' ){
            throw Error(`The first layer must be the input layer, and must be type input!`);
        }

        if( context.input_layer.activation != undefined ){
            throw Error(`The first layer dont need a activation function!`);
        }

        /**
        * Vinculate a prop into this model, to make easy to get and manipulate
        * @param {whatVinculateID}
        * @param {whatVinculate}
        * 
        * @returns {Object} - Model it self
        */
        context.vinculate = function(whatVinculateID, whatVinculate){
            context[ whatVinculateID ] = whatVinculate;
            context.vincules[ whatVinculateID ] = context[ whatVinculateID ];
            return context;
        }

        context.readProp = function( whatReadName ){
            return context[whatReadName];
        }

        /**
        * Get the training type 
        */
        context.getTrainingType = function(){
            return context.training_type;
        }

        /**
        * Get the first hidden layer
        * @returns {Object}
        */
        context.get_first_hidden_layer = function(){
            return context.layers[0];
        }

        /**
        * Getter for the context.layers
        * @returns {Array}
        */
        context.getLayers = function(){
            return context.layers;
        }

        /**
        * Get a layer
        * @param {Number} layer_index 
        * @returns {Object}
        */
        context.getLayer = function( layer_index ){
            return context.layers[ layer_index ];
        }

        /* Get the output layer */
        context.getOutputLayer = function(){
            return context.layers[ context.layers.length-1 ];
        }

        /**
        * Import the weights and bias from context.export
        */
        context.import_from_json = function( nn_saved_structure={} ){

            let number_of_layers = context.layers.length;
        
            if( typeof nn_saved_structure != 'object' ){
                throw Error(` The nn_saved_structure is not a JSON!`);
            }
        
            if( Object.values(nn_saved_structure).length == 0 ){
                throw Error(` The nn_saved_structure is empty JSON!`);
            }
        
            if( nn_saved_structure['layers_data'] == undefined ){
                throw Error(` The nn_saved_structure not have 'layers_data' property!. Invalid object!`);
            }
        
            //If you don't have hair, a layer
            if( nn_saved_structure['layers_data']['layer0'] == undefined ){
                throw Error(` The nn_saved_structure=${nn_saved_structure} not have layers!`);
            }
        
            for( let L = 0 ; L < number_of_layers ; L++ )
            {
                let current_layer_data = context.layers[L];
                let number_of_units    = current_layer_data.units.length;
                let layer_data         = nn_saved_structure.layers_data[`layer${L}`];
        
                //For each unit
                for( let U = 0 ; U < number_of_units ; U++ )
                {
                    //Imported data
                    let imported_current_unit = layer_data[`unit${U}`];
                    let imported_weights      = imported_current_unit.weights;
                    let imported_bias         = imported_current_unit.bias;
        
                    //Model data
                    let model_current_layer = current_layer_data;
                    let model_current_unit  = current_layer_data.units[U];
        
                    //Set imported data
                    model_current_unit.setWeights( imported_weights );
                    model_current_unit.setBias( imported_bias );
                }
            }
        }

        /**
        * Save the weights and bias in a JSON object
        */
        context.export = function(){
            
            let nn_saved_structure = {
                'layers_data': {}
            };
            let number_of_layers = context.layers.length;

            nn_saved_structure['number_of_layers'] = number_of_layers;

            for( let L = 0 ; L < number_of_layers ; L++ )
            {
                nn_saved_structure.layers_data[`layer${L}`] = {};

                let current_layer_data = context.layers[L];
                let number_of_units = current_layer_data.units.length;

                for( let U = 0 ; U < number_of_units ; U++ )
                {
                    let U_data = {
                        weights: current_layer_data.units[U].weights,
                        bias: current_layer_data.units[U].bias
                    }

                    nn_saved_structure.layers_data[`layer${L}`][`unit${U}`] = U_data;
                }
            }

            nn_saved_structure['number_of_inputs'] = nn_saved_structure.layers_data.layer0.unit0.weights.length;
            nn_saved_structure['number_of_outputs'] = Object.keys( nn_saved_structure.layers_data[`layer${number_of_layers-1}`] ).length;

            return nn_saved_structure;
        }

        //store the initial weights and biases
        context.initial_weights = context.export();
    }

}