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
}

//Activation functions
net.activations = {};

net.activations.sigmoid = function(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

//Just the derivative of sigmoid
net.activations.sigmoid.derivative = function(functionOutput){
    return functionOutput * (1.0 - functionOutput);
}

net.activations.relu = function(x) {
    return Math.max(0, x);
}

//Just the derivative of sigmoid
net.activations.relu.derivative = function(functionOutput){
    return functionOutput > 0 ? 1 : 0;
}

/**
* A utility for manipulate weights
* @param {Object} config 
*/
net.WeightManipulator = function( config ){
    let context = {};
    context._unit = config['context'];
    context.weightID = config['index'];
    context.value = context._unit.getWeight(context.weightID);
    context.input = context._unit.getInputOfWeight(context.weightID);

    context.add = function( number ){
        context._unit.weights[ context.weightID ] = context._unit.weights[ context.weightID ] + number;
    }

    context.subtract = function( number ){
        context._unit.weights[ context.weightID ] = context._unit.weights[ context.weightID ] - number;
    }

    context.reset = function(){
        context._unit.weights[ context.weightID ] = 0;
    }

    return context;
}

/**
* A utility used for accumulate the LOSSES in a hidden layer
* This accumulation is made using a sum 
*/
net.LOSS_Accumulator = function(){
    let context = {};
    context._acumulated = 0;

    /** 
    * Compute the derivative using the chain rule AND sum in the __acumulated 
    * 
    * @param {Number} eloh_param  - The parameters what connect the unit
    * @param {Number} LOSS        - The unit LOSS
    */
    context.accumulate = function( eloh_param=Number(), 
                                   LOSS=Number() 
    ){
        let thisDerivative  = eloh_param * LOSS;
        context._acumulated = context._acumulated + thisDerivative;
    }

    /**
    * Get the accumulated value
    * @returns {Number} - the accumulated value
    */
    context.getAccumulatedValue = function(){
        return context._acumulated;
    }

    return context;
}

//A Unit(with just feedforward and weight initialization)
net.Unit = function( unit_config={} ){
    let context = {};

    //Parameters
    context.number_of_inputs      = unit_config.number_of_inputs     || Number();
    context.activation_function   = unit_config.activation_function  || 'sigmoid';
    context.weights               = unit_config.weights              || Array(context.number_of_inputs).fill(0);
    context.bias                  = unit_config.bias                 || Number();

    /**
    * Get the layer object( That was linked to this unit )
    * @returns {Object}
    */
    context.getLayerOfThisUnit = function(){
        return context._layerRef; 
    }

    /**
    * Vinculate a prop to this unit
    * @param {String} newAttributeName
    * @param {any}    valueOfThisAttribute
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
    }

    /**
    * Get the activation function name
    */
    context.getFunctionName = function(){
        return context.activation_function;
    }

    //Setters
    context.setWeights = function( newWeights=[] ){
        if( !(newWeights instanceof Array) ){
            throw Error(`the newWeights=${newWeights} is not a Array instance!`);
        }
        if( newWeights.length == 0 ){
            throw Error(`the newWeights is empty Array!`);
        }
        context.weights = newWeights;
    }

    //Getter for the context.weights
    context.getWeights = function(){
        return context.weights;
    }

    //Add a value to a especific weight
    context.addWeight = function( weight_index, number ){
        context.weights[ weight_index ] = context.weights[ weight_index ] + number;
    }

    //Subtract a value to a especific weight
    context.subtractWeight = function( weight_index, number ){
        context.weights[ weight_index ] = context.weights[ weight_index ] - number;
    }

    //Get a WeightManipulator for a especific weight
    context.selectWeight = function( weight_index ){
        return net.WeightManipulator({
            context: context,
            index: weight_index
        });
    }

    //Setter for bias
    context.setBias = function( newBias=Number() ){
        if( typeof newBias != 'number' ){
            throw Error(`the newBias=${newBias} is not a Number instance!`);
        }
        context.bias = newBias;
    }

    //Add a value for the bias
    context.addBias = function( number ){
        context.bias = context.bias + number;
    }

    //Subtract a value for the bias
    context.subtractBias = function( number ){
        context.bias = context.bias - number;
    }

    /**
    * Generate random parameters
    */
    context.generate_random_parameters = function( number_of_inputs=context.number_of_inputs ){

        for( let i = 0 ; i < number_of_inputs ; i++ )
        {
            context.weights[i] = Math.random();
        }

        context.bias = Math.random();
    }

    /**
    * Do the feedforward step for ONE SAMPLE of this UNIT, for estimate output
    * @param   {Array}   sample_inputs        - The model inputs
    * @returns {Number}                       - The model estimative
    */
    context.estimateOutput = function( sample_inputs )
    {
        let number_of_inputs = sample_inputs.length;

        let summed_value = 0;

        for( let i = 0 ; i < number_of_inputs ; i++ ) 
        {
            let weight_index = i;
            summed_value = summed_value + ( sample_inputs[i] * context.getWeight( weight_index ) );
        }

        //Add the bias
        summed_value = summed_value + context.bias;

        let output = net.activations[ context.getFunctionName() ]( summed_value );

        return {
            activation_function_output: output,
            unit_potential: summed_value //The unit activation potential(just the summed value)
        };
    }

    /**
    * Get a especific weight of this unit 
    * @param {Number} weight_index
    * @returns {Number}
    */
    context.getWeight = function( weight_index ){
        return context.weights[ weight_index ]
    }

    /**
    * Get a especific input of weight
    * @param {Number} weight_index 
    * @returns {Number}
    */
    context.getInputOfWeight = function( weight_index ){
        let father_layer          = context.getLayerOfThisUnit(), //Get the layer to which THIS UNIT belongs
            father_layer_inputs   = father_layer.getInputs();     //The inputs of the layer(that is, the layer to which THIS UNIT belongs)

        return father_layer_inputs[ weight_index ];
    }

    return context;
}

//A Layer
net.Layer = function( layer_config={} ){
    let context = {};

    context.layer_config          = layer_config;
    context.number_of_units       = layer_config.units;
    context.number_of_outputs     = context.number_of_units; //the same as context.number_of_units
    context.number_of_inputs      = layer_config.inputs;
    context.activation_function   = layer_config.activation;
    context.layer_type            = layer_config.type;

    //The units objects that will be created below
    context.units        = [];

    //Initialize the layer
    for( let i = 0 ; i < context.number_of_units ; i++ )
    {
       //Add a unit to the list
       context.units[i] = net.Unit({
           number_of_inputs     : context.number_of_inputs,
           activation_function  : context.activation_function
       });

       context.units[i].vinculate( '_unitIndex',  i       );
       context.units[i].vinculate( '_layerRef' ,  context );
       context.units[i].generate_random_parameters( context.number_of_inputs );
    }

    /**
    * Vinculate a prop to this layer
    * @param {String} newAttributeName
    * @param {any}    valueOfThisAttribute
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
    }

    /**
    * Get the layer number(in model) 
    */
    context.getIndex = function(){
        return context['_internal_index'];
    }

    /**
    * Get the layer father(the model)
    */
    context.getFather = function(){
        return context._father;
    }

    /**
    * Return the next layer( That is, it returns the layer that comes after the current layer )
    * @returns {net.Layer}
    */
    context.getNextLayer = function(){
        let next_layer_index = context.getIndex() + 1;

        return context.getFather()
                      .getLayer( next_layer_index ) || null;
    }

    /**
    * Return the previous layer( That is, it returns the layer that comes before the current layer )
    * @returns {net.Layer}
    */
    context.getPreviousLayer = function(){
        let previous_layer_index = context.getIndex() - 1;

        return context.getFather()
                      .getLayer( previous_layer_index ) || null;
    }

    /**
    * Check if this layer is of type 
    */
    context.is = function( layerType ){
        return context.layer_type == layerType ? true : false;
    }

    /**
    * Check if this layer NOT IS of type 
    */
    context.notIs = function( layerType ){
        return !context.is( layerType );
    }

    /**
    * A getter for context.units
    */
    context.getUnits = function(){
        return context.units;
    }

    /**
    * Get a unit in this layer
    * @param {Number} unit_index 
    * @returns {Object}
    */
    context.getUnit = function( unit_index ){
        return context.units[ unit_index ];
    }

    /**
    * Set the inputs of this layer, to be used in context.get_Output_of_Units
    */
    context.setInputs = function( LAYER_INPUTS=[] ){
        let this_layer_ref        = context,
            this_layer_index      = this_layer_ref.getIndex(),
            father_ref            = this_layer_ref.getFather(),
            inputs_of_each_layer  = father_ref.readProp('inputs_of_each_layer');  

        if( !(LAYER_INPUTS instanceof Array) ){
            throw Error(`LAYER_INPUTS need be a Array!`);
        }

        /**
        * Assign the LAYER_INPUTS in the prop "inputs_of_each_layer"
        */
        inputs_of_each_layer[ `layer${ this_layer_index }` ] = [... LAYER_INPUTS.copyWithin()];

        /*
        * Create a direct reference to make easy access this via layer
        */
        this_layer_ref[ 'LAYER_INPUTS' ] = inputs_of_each_layer[ `layer${ this_layer_index }` ];
    }

    /**
    * Get the inputs of this layer, to be used in context.get_Output_of_Units
    */
    context.getInputs = function(){
        return context['LAYER_INPUTS'];
    }

    /** 
    * Get the output of each unit in the current layer 
    */
    context.get_Output_of_Units = function(){

        /**
        * Get this layer inputs( that was linked to this object )
        * Remembering that the inputs of this layer are the outputs of the previous layer
        */
        let LAYER_INPUTS = context.getInputs();

        /**
        * For each unit in this layer <layer_index>, get the UNIT OUTPUT and store inside the unit
        */
        let units_outputs = [];

        /**
        * Compute the output of each unit in this layer 
        */
        context.getUnits().forEach(function( current_unit ){

            let unit_output_data  = current_unit.estimateOutput( LAYER_INPUTS );
            let act_potential     = unit_output_data.unit_potential;
            let unit_output       = unit_output_data.activation_function_output;

            units_outputs.push( unit_output );
        });

        return units_outputs;
    }

    /**
    * Return the layer ready
    */
    return context;
}

//The entire model
net.MLP = function( config_dict={} ){
    let context = {};

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
    let last_created = null;

    //Initialize the network (ignoring the input layer)
    for( let i = 1 ; i < context.number_of_layers ; i++ )
    {
        let current_layer = net.Layer( context.layers_structure[i] );

        /**
        * Here I used "i-1" precisely because we are ignoring the input layer, as this for loop starts at layer 1 forward (precisely to ignore the input layer)
        */
        current_layer.vinculate('_internal_index', i-1);

        current_layer.vinculate('_father',         context);

        context.layers.push( current_layer );


        //Validations of layer creation
        if( last_created && current_layer.number_of_inputs != last_created.number_of_units ) {
            throw Error(`Initialization error: The layer ${i} have ${ current_layer.number_of_inputs } inputs. But, should be ${ last_created.number_of_units } inputs, because the previus layer( the layer ${i-1} ) have ${ last_created.number_of_units } output units.`);
        }
        last_created = current_layer;
    }


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
    * @parma {whatVinculate}
    */
    context.vinculate = function(whatVinculateID, whatVinculate){
        context[ whatVinculateID ] = whatVinculate;
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

    /**
    * Feedforward for a ONE SAMPLE
    * @param {Array} sample_inputs 
    * @returns {Array}
    */
    context.feedforward_sample = function( sample_inputs=[] ){

        //Validations
        if( !(sample_inputs instanceof Array) ){
            throw Error(`The sample_inputs=${sample_inputs} need be a Array instance!`);
        }

        if( sample_inputs.length == 0 ){
            throw Error(`The sample_inputs is empty Array!`);
        }

        /**
        * Store the inputs of each layer (that all units of the layer receives)
        * And vinculate this object in the MLP(the father of the layers) to easy access and manipulations
        */
        let inputs_of_each_layer   = {};
        context.vinculate('inputs_of_each_layer', inputs_of_each_layer);

        /**
        * Store the ouputs of each unit of each layer 
        * And vinculate this object in the MLP(the father of the layers) to easy access and manipulations
        */
        let outputs_of_each_layer  = {};
        context.vinculate('outputs_of_each_layer', outputs_of_each_layer);

        /**
        * The inputs of a layer <layer_index> is always the outputs of previous layer( <layer_index> - 1 )
        * So, the property LAYER_INPUTS of the first hidden layer is the sample_inputs. And in the feedforward, each layer(<layer_index>) will have the property LAYER_INPUTS, storing the outputs of the previous layer(<layer_index> - 1) 
        * 
        * So, the inputs of first hidden layer( that is <layer_index>=0 ), will be the sample_inputs
        * And the inputs of secound hidden layer( that is <layer_index>=1 ), will be the outputs of the first hidden layer( that is <layer_index>=0 )
        *
        * Always in this way.
        */
        context.get_first_hidden_layer()
               .setInputs( [... sample_inputs] );

        //The outputs of OUTPUT LAYER
        let final_outputs         = []; 

        /**
        * In this case, the layer 0 is the first hidden layer, because the input layer is ignored in initialization
        *
        * So
        * For each layer:
        */
        context.getLayers().forEach(function( current_layer, 
                                              layer_index 
        ){
            /**
            * For each unit in current layer, get the UNIT OUTPUT and store inside the unit
            */
            let units_outputs = current_layer.get_Output_of_Units();

            /**
            * Store the outputs
            */
            outputs_of_each_layer[ `layer${ layer_index }` ] = {};
            units_outputs.forEach(function( unitOutput, index ){
                outputs_of_each_layer[ `layer${ layer_index }` ][ `unit${ index }` ] = unitOutput;
            });
            
            /**
            * If the current layer is NOT the output layer
            */
            if( current_layer.notIs('output') ){

                /*
                * The inputs of a layer <layer_index> is always the outputs of previous layer( <layer_index> - 1 ) 
                * Then the in lines below will Store the outputs of the current layer( <layer_index> ) in the NEXT LAYER( <layer_index> + 1 ) AS UNIT_INPUTS
                */
                let next_layer = current_layer.getNextLayer();
            
                /**
                * Set the current layer( <layer_index> ) outputs AS UNIT_INPUTS OF THE NEXT LAYER( <layer_index> + 1 )
                */
                next_layer.setInputs( units_outputs );
            }

            /**
            * If is the output layer
            */
            if( current_layer.is('output') )
            {
                final_outputs = units_outputs;
            }

        });

        /**
        * Return the final outputs( that is the outputs of the output layer )
        */
        return {
            output_estimated_values : final_outputs,
            inputs_of_each_layer    : inputs_of_each_layer,
            outputs_of_each_layer   : outputs_of_each_layer,

            /**
            * Get the estimated outputs
            * @returns {Array}
            */
            getEstimatedOutputs: function(){
                return final_outputs;
            },

            /**
            * Get the input of each layer
            * @returns {Array}
            */
            getInputsOfEachLayer: function(){
                return inputs_of_each_layer;
            },

            /**
            * Get the output of each layer
            * @returns {Array}
            */
            getOutputsOfEachLayer: function(){
                return outputs_of_each_layer;
            }
        };
    }

    /* Get the output layer */
    context.getOutputLayer = function(){
        return context.layers[ context.layers.length-1 ];
    }

    /**
    * Calculate the derivative of each unit in the output layer
    * 
    * @param {Array}  output_estimated_values              - The network estimations(of each unit)
    * @param {Array}  desiredOutputs                       - The desired outputs(for each unit)
    * @param {Object} list_to_store_gradients_of_units     - A object to store the gradients of each unit
    * @param {Object} list_to_store_gradients_for_weights  - A object to store the gradients of each unit with respect to each unit weight
    *
    * @returns {Object} - The calculated gradients of the output layer
    * 
    */
    context.calculate_derivatives_of_output_units = function( output_estimated_values, 
                                                              desiredOutputs, 
                                                              list_to_store_gradients_of_units, 
                                                              list_to_store_gradients_for_weights 
    ){

        let number_of_layers         = context.getLayers().length;
        let index_of_output_layer    = number_of_layers-1;

        //Registry a object for store the gradients of the units of the output layer
        list_to_store_gradients_of_units[ `layer${ number_of_layers-1 }` ] = {};
        list_to_store_gradients_for_weights[ `layer${ number_of_layers-1 }` ] = {};

        /**
        * Calculate the LOSS derivative of each output unit, using the chain rule of calculus
        * That is, the derivative of the LOSS with respect to estimated value of the output unit
        * 
        * For calculate the LOSS derivative of the each output unit, The formula used is:
        *  
        *   unit<N>_derivative = ( output_unit<N>_estimated_value - output_unit<N>_desired_value ) * derivative_of_function_of_output_unit<N>
        *
        * Like he can see below:
        */
        context.getOutputLayer().getUnits().forEach(function( output_unit, 
                                                              output_unit_index 
        ){

            let unitActivationFn     = output_unit.getFunctionName();
            
            let unitOutput           = output_estimated_values[ output_unit_index ];
            
            let desiredOutput        = desiredOutputs[ output_unit_index ];

            let outputDifference     = unitOutput - desiredOutput;
            
            //The activation function of this U output unit
            let unit_function_object = net.activations[ unitActivationFn ];

            //The derivative of activation funcion of this U output unit(at output layer)
            let outputDerivative  = unit_function_object.derivative( unitOutput );

            //The derivative of this output unit U
            let unit_derivative   = outputDifference * outputDerivative;

            //Store the gradient in the gradients object
            list_to_store_gradients_of_units[ `layer${ index_of_output_layer }` ][ `unit${ output_unit_index }` ] = unit_derivative;

            //Store the gradient with respect of each weight
            list_to_store_gradients_for_weights[ `layer${ index_of_output_layer }` ][ `unit${ output_unit_index }` ] = [];
            
            //For each weight
            output_unit.getWeights().forEach(function(weight_value, weight_index_c){
            
                let weight_input_C = output_unit.getInputOfWeight( weight_index_c );  
                list_to_store_gradients_for_weights[ `layer${ index_of_output_layer }` ][ `unit${ output_unit_index }` ][ weight_index_c ] = unit_derivative * weight_input_C;
            
            });

        });

        return {
            calculated_gradients_of_units: {... JSON.parse(JSON.stringify(list_to_store_gradients_of_units)) },
            calculated_gradients_for_weights: {... JSON.parse(JSON.stringify(list_to_store_gradients_for_weights)) }
        };
    }

    /**
    * Calculate a derivative of a especific unit in a hidden layer
    * 
    * @param {Number} index_of_current_hidden_layer      - The index of the current hidden layer( Who owns the UH unit(that we are calculating the derivative) )
    * @param {Number} current_hidden_unit_index          - The index of the UH unit(that we are calculating the derivative)
    * @param {Number} weights_of_current_hidden_unit     - The weights of the UH unit(that we are calculating the derivative)
    * @param {Array}  current_unit_inputs_values         - The inputs of of the UH unit(that we are calculating the derivative)
    * @param {String} current_unit_function_name         - The function name of the UH unit(that we are calculating the derivative)
    * @param {Number} current_unit_output_value          - The output of the UH unit(that we are calculating the derivative)
    * @param {Array}  next_layer_units_objects           - The units of the next layer
    * @param {Object} next_layer_units_gradients         - The gradients of the all units in the next layer
    *
    * @param {Object} map_to_store_gradients_of_units    - The list to store the calculated gradients of the UH unit(that we are calculating the derivative)
    * @param {Object} map_to_store_gradients_for_weights - The list to store the calculated gradients with respect each weight of the UH unit(that we are calculating the derivative)
    * 
    * @returns {Number} - the derivative of the unit
    */
    context.calculate_hidden_unit_derivative = function( index_of_current_hidden_layer=Number(), 
                                                         current_hidden_unit_index=Number(), 
                                                         weights_of_current_hidden_unit=Array(),
                                                         current_unit_inputs_values=Array(), 
                                                         current_unit_function_name=String(), 
                                                         current_unit_output_value=Number(), 
                                                         next_layer_units_objects=Array(), 
                                                         next_layer_units_gradients={}, 

                                                         //List to store the values
                                                         map_to_store_gradients_of_units={}, 
                                                         map_to_store_gradients_for_weights={}
    ){

        let unit_function_object      = net.activations[ current_unit_function_name ];
    
        /*
        * THE FORMULA USED IS FOLLOWING:
        *
        * The gradients for the units in ANY hidden layer is:
        * Below are a simple example suposing that in the next layer we have 2 units:
        * 
        * >>> EQUATION WITH A EXAMPLE OF USE:
        * 
        *    current_unit<UH>_derivative  = (next_layer_unit<N0>.weight<UH> * derivative_of_next_layer_unit<N0>) + 
        *                                   (next_layer_unit<N1>.weight<UH> * derivative_of_next_layer_unit<N1>) + 
        *                                   [... etc]
        * 
        *    NOTE: In this example, the next layer have just 2 units(N0 and N1, respectively), 
        *          but There could be as many as there were. By this, i put "[... etc]", to make it clear that there could be more than just 2 units
        * 
        *    NOTE: "current_unit" is a hidden unit!
        * 
        * 
        * >>> EXPLANATION:
        * 
        *   Where the UH is the index of the hidden unit in the current hidden layer. And the N is the index of the next layer unit.
        *   Relemering that the in the example above, we have just 2 units in the next layer, so the have only the N0(unit one) and N1(unit two).
        * 
        *   The weight<UH> is the connection weight of weights array in the next_layer_unit<N> object.
        * 
        * 
        * This is the equation that are used for apply the backpropagation. This equation is used in this loop.
        * So the code bellow apply this:
        * 
        */

        // Do the accumulation of the LOSSES in the next layer
        let current_unit_accumulator = net.LOSS_Accumulator();

        /**
        * For each unit in LEXT LAYER 
        * We are working in the context of the next layer:
        * 
        *    Here "unit" is the current unit in the next layer(that is, current of the forEach loop)
        *   "unit_index" is the number of the current unit in the next layer
        */
        next_layer_units_objects.forEach(function( unit, 
                                                   unit_index 
        ){

            let connection_weight_with_UH   = unit.getWeight( current_hidden_unit_index );
            
            let derivative_of_unit          = next_layer_units_gradients[ `unit${ unit_index }` ];

            /**
            * NOTE: The next_layer_unit_N.weights[ UH ] is the connection weight, whose index is UH(of the external loop in the explanation of the equation above)
            *       Because, for example, if we are calculating the gradient of the first unit in the last hidden layer, 
            *       These gradient(of the hidden unit) will depedent of the all gradients in the output layer, 
            *       together with the connection weight, that is, the weight of unit N of the output layer with respect to the hidden unit number UH
            *
            * Above are the gradient equation for the hidden layer units, that are applied in the line below:
            */

            current_unit_accumulator.accumulate( 
                                    eloh_param  = connection_weight_with_UH,
                                    LOSS        = derivative_of_unit 
                                );
        });

        let acumulated      = current_unit_accumulator.getAccumulatedValue();

        let unit_derivative = acumulated * unit_function_object.derivative( current_unit_output_value );
        
        /**
        * Store the gradient in gradients object
        */
        map_to_store_gradients_of_units[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ] = unit_derivative;

        /*
        * Aditionally, store the erros TOO with respect of each weight
        */
        map_to_store_gradients_for_weights[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ] = [];
        
        weights_of_current_hidden_unit.forEach(function( weight_value, 
                                                         weight_index_c
        ){

            let weight_input_C = current_unit_inputs_values[ weight_index_c ]; //CRIAR UM GETTER  
            map_to_store_gradients_for_weights[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ][ weight_index_c ] = unit_derivative * weight_input_C;
        
        });

        //Return the actual gradients
        return {
            map_to_store_gradients_of_units     : map_to_store_gradients_of_units,
            map_to_store_gradients_for_weights  : map_to_store_gradients_for_weights
        }
    }

    /**
    * Do the backpropagation step for ONE SAMPLE 
    * 
    * This backpropagation implementation was inspired by various publicly available resources, including concepts 
    * and explanations from the work of Jason Brownlee on backpropagation.
    * 
    * CREDITS && REFERENCE:
    * Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
    * 
    *
    * @param {Array} sample_inputs  - the sample features
    * @param {Array} desiredOutputs - the DESIRED outputs of the output units
    * 
    * @returns {Object} - The mapped gradients of the units of each layer AND The mapped gradients of the each weight of each unit of each layer
    */
    context.backpropagate_sample = function( sample_inputs  = [], 
                                             desiredOutputs = [] 
    ){
        
        //Validations
        if( !(sample_inputs instanceof Array) ){
            throw Error(`The sample_inputs=[${sample_inputs}] need be a Array instance!`);
        }

        if( sample_inputs.length == 0 ){
            throw Error(`The sample_inputs is empty Array!`);
        }

        if( !(desiredOutputs instanceof Array) ){
            throw Error(`The desiredOutputs=[${desiredOutputs}] need be a Array instance!`);
        }

        if( desiredOutputs.length == 0 ){
            throw Error(`The desiredOutputs is empty Array!`);
        }

        let number_of_layers         = context.getLayers().length;
        
        //Do the feedforward step
        let feedforward_data         = context.feedforward_sample( sample_inputs );

        //Extract the data from the feedforward
        let output_estimated_values  = feedforward_data.getEstimatedOutputs();
        let inputs_of_each_layer     = feedforward_data.getInputsOfEachLayer();
        let outputs_of_each_layer    = feedforward_data.getOutputsOfEachLayer();
         
        /**
        * List store the gradients of each unit of the all layers
        * Format { layer_number: gradients_object ...}
        */
        let list_to_store_gradients_of_units = {};

        /**
        * List store the gradients for the all layers(of each weight of each unit)
        * Format { layer_number: [ unit: { weight: [] ...} ] ...}
        */
        let list_to_store_gradients_for_weights = {};

        /**
        * Calculate the derivative of each output unit
        * This process is made by a subtraction of the "unit estimated value" and the "desired value for the unit".
        * So, Each unit have a "desired value", and each unit produces a "estimative" in the feedforward phase, so these informations are used to calculate this derivatives
        * 
        * And these gradients will be stored in the list_to_store_gradients_of_units and list_to_store_gradients_for_weights
        */
        context.calculate_derivatives_of_output_units( output_estimated_values, 
                                                       desiredOutputs, 
                                                       list_to_store_gradients_of_units, 
                                                       list_to_store_gradients_for_weights );
        
        /** 
        * Start the backpropagation
        * Starting in LAST HIDDEN LAYER and going in direction of the FIRST HIDDEN LAYER.
        * 
        * When we reach the FIRST HIDDEN LAYER, when had calculate the gradients of all units in the FIRST HIDDEN LAYER, the backpropagation finally ends.
        */
        let currentLayerIndex = number_of_layers-1-1;

        /**
        * While the "while loop" not arrived the first hidden layer
        * The first hidden layer(that have index 0, will be the last layer that will be computed) 
        */
        while( currentLayerIndex >= 0 )
        {
            //Current layer data
            let current_layer          = context.getLayer( currentLayerIndex );
            let current_layer_inputs   = inputs_of_each_layer[ `layer${ currentLayerIndex }` ];
            let current_layer_outputs  = outputs_of_each_layer[ `layer${ currentLayerIndex }` ]

            list_to_store_gradients_of_units[ `layer${ currentLayerIndex }` ]     = {};
            list_to_store_gradients_for_weights[ `layer${ currentLayerIndex }` ]  = {};

            /**
            * Next layer data:
            */
            let next_layer_index               = currentLayerIndex + 1;
            let next_layer                     = context.getLayer( next_layer_index );
            let next_layer_units               = next_layer.getUnits();

            /**
            * Get the gradients(of the units) of the next layer
            * These gradients will be used in lines below:
            */
            let next_layer_gradients           = list_to_store_gradients_of_units[ `layer${ next_layer_index }` ];

            /**
            * For each unit in CURRENT HIDDEN LAYER
            */
            current_layer.getUnits().forEach(function( current_hidden_layer_unit, 
                                                       the_unit_index
            ){

                let hidden_unit_index           = the_unit_index; //I also will call as UH, that is The index of the current unit, like in the equation above;
                let current_unit_weights        = current_hidden_layer_unit.getWeights();
                let current_unit_output         = current_layer_outputs[ `unit${ the_unit_index }` ];
                let current_unit_function_name  = current_hidden_layer_unit.getFunctionName();
                
                /**
                * I make a copy of the "current_layer_inputs" variable to make a safe and independent copy of the same layer inputs for all units,
                * 
                * Because, the key point of this is: 
                * 
                *    All units receives the same inputs!. That is, the same inputs of the layer Who owns the unit
                *    Because, each layer have N inputs. And the inputs of EACH UNIT in a layer are the outputs of the previous layer.
                *    So, in short, in a given layer of the neural network, all units in that layer will receive exactly the same inputs, THAT IS, THE OUTPUT OF THE PREVIOUS LAYER, WHICH ARE ITS INPUTS   
                *
                *    Except the input layer, because the input layer has no units, It only has inputs(just numbers), and nothing more than that.
                *    Therefore, the input layer has no units, and therefore does not receive input from a previous layer
                *
                *    To facilitate and standardize the process, in a generic way for each layer, we can imagine that the outputs of the previous layer are the inputs themselves.               
                */
                let current_unit_inputs       = [... current_layer_inputs.copyWithin()];

                /**
                * Calculate the derivative of the current unit UH
                * Relembering that, 
                * The derivative of a unit in a hidden layer will always depend of the derivatives of the next layer
                */ 
                context.calculate_hidden_unit_derivative( 
                                                        index_of_current_hidden_layer        = currentLayerIndex,
                                                        current_hidden_unit_index            = hidden_unit_index, 
                                                        weights_of_current_hidden_unit       = current_unit_weights,
                                                        current_unit_inputs_values           = current_unit_inputs,
                                                        current_unit_function_name           = current_unit_function_name,
                                                        current_unit_output_value            = current_unit_output,
                                                        next_layer_units_objects             = next_layer_units,
                                                        next_layer_units_gradients           = next_layer_gradients,

                                                        list_to_store_gradients_of_units     = list_to_store_gradients_of_units,
                                                        list_to_store_gradients_for_weights  = list_to_store_gradients_for_weights
                                                    );
            });

            /**
            * Goto previous layer
            */
            currentLayerIndex--;
        }

        /**
        * Return the calculated gradients for the sample
        */
        return {
            gradients_of_units          : list_to_store_gradients_of_units,
            gradients_for_each_weights  : list_to_store_gradients_for_weights
        };
    }

    /**
    * Applies the Gradient Descent algorithm 
    * Is used to update the weights and bias of each unit in each layer
    * 
    * @param {Object} the_gradients_for_weights - The gradients of each weight of each unit of each layer
    * @param {Object} the_gradients_for_bias    - The gradients of the bias of each unit of each layer
    */
    context.optimize_the_parameters = function( the_gradients_for_weights={}, 
                                                the_gradients_for_bias={} 
    ){
        //For each layer
        context.getLayers().forEach(function( current_layer, 
                                              layer_index                                       
        ){
            
            //For each unit in current layer
            current_layer.getUnits().forEach(function( current_unit, 
                                                       unit_index
            ){

                //For each weight
                current_unit.getWeights().forEach(function( weight_value, 
                                                            weight_index
                ){

                    let calculated_gradients_values_for_weight = the_gradients_for_weights[`layer${ layer_index }`][ `unit${ unit_index }` ][ weight_index ];

                    //Select this weight <weight_index> and update then
                    current_unit.selectWeight( weight_index )
                                .subtract( context.learning_rate * calculated_gradients_values_for_weight );
                
                });

                let calculated_gradients_values_for_bias = the_gradients_for_bias[`layer${ layer_index }`][ `unit${ unit_index }` ];

                //Update bias
                current_unit.subtractBias( context.learning_rate * calculated_gradients_values_for_bias );

            });

        });
    }

    /**
    * Compute de COST
    * 
    * @param {Array} train_samples - The training samples
    * @returns {Number} - The cost
    */
    context.compute_train_cost = function( train_samples ){

        //Validations
        //If is a array or a instance of net.data.Dataset, is acceptable and compative type
        if( !(train_samples instanceof Array || train_samples.internalType == 'dataset') ){
            throw Error(`The train_samples=[${train_samples}] need be a Array instance!`);
        }

        if( train_samples.length == 0 ){
            throw Error(`The train_samples is a empty Array!`);
        }

        let cost = 0;
        
        train_samples.forEach(function( sample_data ){  

            let sample_features        = sample_data[0]; //SAMPLE FEATURES
            let sample_desired_value   = sample_data[1]; //SAMPLE DESIRED OUTPUTS
            let estimatedValues        = context.feedforward_sample(sample_features).getEstimatedOutputs();

            for( let S = 0 ; S < estimatedValues.length ; S++ )
            {
                cost = cost + ( sample_desired_value[ S ] - estimatedValues[ S ] ) ** 2;
            }

        });

        return cost;
    }

    /**
    * SGD/Online training
    * Update the weights after each individual example
    * 
    * @param {Array} train_samples 
    * @param {Array} number_of_epochs 
    * @returns {Object}
    */
    context.online_train = function( train_samples, 
                                     number_of_epochs
    ){
        let currentEpoch    = 0;
        let last_total_loss = 0;
        let loss_history    = [];

        //While the current epoch number is less the number_of_epochs
        while( currentEpoch < number_of_epochs )
        {
            let total_loss = 0;

            //For each sample
            train_samples.forEach(function( sample_data ){

                let sample_features         = sample_data[0]; //SAMPLE FEATURES
                let sample_desired_value    = sample_data[1]; //SAMPLE DESIRED OUTPUTS

                //Validations before apply the backpropagation
                if( !(sample_features instanceof Array) ){
                    throw Error(`The variable sample_features=[${sample_features}] must be a Array!`);
                }

                if( !(sample_desired_value instanceof Array) ){
                    throw Error(`The variable sample_desired_value=[${sample_desired_value}] is not a Array!`);
                }

                //If the number of items in the sample_desired_value Array is different from the number of units in the output layer
                if( sample_desired_value.length != context.layers[ context.layers.length-1 ].units.length ){
                    throw Error(`The sample_desired_value=[${sample_desired_value}] has ${sample_desired_value.length} elements, But must be ${context.layers[ context.layers.length-1 ].units.length}(the number of units in output layer)`);
                }

                //Do backpropagation and Gradient Descent
                let calculated_gradients_data         = context.backpropagate_sample(sample_features, sample_desired_value);
            
                let gradients_for_weights  = calculated_gradients_data['gradients_for_each_weights']
                let gradients_for_bias     = calculated_gradients_data['gradients_of_units']

                /**
                * Applies the Gradient Descent algorithm to update the parameters
                */
                context.optimize_the_parameters( gradients_for_weights, gradients_for_bias );

            });

            total_loss += context.compute_train_cost( train_samples );

            last_total_loss = total_loss;
            loss_history.push(total_loss);
            
            if( String( Number(currentEpoch / 100) ).indexOf('.') != -1 ){
                console.log(`LOSS: ${last_total_loss}, epoch ${currentEpoch}`)
            }

            //Goto next epoch
            currentEpoch++;
        }

        return {
            last_total_loss: last_total_loss,
            loss_history: loss_history
        };
    }

    /**
    * Acculumate the gradients of a full-batch(that is, of each sample in training_samples), and update the parameters
    * This method will be used in the "fullbatch_train" training metheod
    * 
    * @param {Array} train_samples - The training samples
    */
    context.train_fullbatch = function( train_samples ){
        let number_of_samples = train_samples.length;

        /**
        * The variable summed_gradients_for_weights, is used for:
        * Accumulate the gradient of each weight of each unit of each layer
        * 
        * Structure in visually representation is:
        *       layer0
        *          --> unit0
        *              --> accumulation_for_weight_0
        *              --> accumulation_for_weight_1
        *              --> accumulation_for_weight_N
        *                  (etc.. other accumulations)
        *                  (all the "accumulation_for_weight_N" is a Number) 
        * 
        *              (etc... other units)
        * 
        *      (etc... other layers)
        * 
        * Text description of this representation:
        * The variable summed_gradients_for_weights is a hashmap with layers.
        * Each layer have N units, and each unit have N "weight accumulation" or also called "accumulation_for_weight" in this text example, and are Numbers.
        * 
        * This is a accumulation of the gradient of each weight of each unit of each layer
        * The accumulation will be done in the lines bellow, using some for loops:
        */
        let summed_gradients_for_weights = {};

        /**
        * The variable summed_gradients_for_weights, is used for:
        * Accumulate the gradient of the bias of each unit of each layer
        * 
        * Structure in visually representation is:
        *       layer0
        *          --> accumulation_for_bias_of_unit_0  
        *          --> accumulation_for_bias_of_unit_1         
        *           (etc.. other weights gradients)
        *           (all the "bias_of_unit<N>" is a Number) 
        *
        *      (etc... other layers)
        * 
        * Text description of this representation:
        * The variable summed_gradients_for_bias is a hashmap with layers.
        * Each layer have N "bias accumulation"(corresponding to each unit) or also called "accumulation_for_bias_of_unit_<N>" in this text example, and is a Number.
        * 
        * This is a accumulation of the gradient of each the bias of each unit of each layer
        * The accumulation will be done in the lines bellow, using some for loops:
        */
        let summed_gradients_for_bias = {};

        /**
        * For each sample 
        * Will accumulating the gradients(of all samples)
        */
        train_samples.forEach(function( sample_data ){

            let sample_features         = sample_data[0]; //SAMPLE FEATURES
            let sample_desired_value    = sample_data[1]; //SAMPLE DESIRED OUTPUTS

            //Validations before apply the backpropagation
            if( !(sample_features instanceof Array) ){
                throw Error(`The variable sample_features=[${sample_features}] must be a Array!`);
            }

            if( !(sample_desired_value instanceof Array) ){
                throw Error(`The variable sample_desired_value=[${sample_desired_value}] is not a Array!`);
            }

            //If the number of items in the sample_desired_value Array is different from the number of units in the output layer
            if( sample_desired_value.length != context.layers[ context.layers.length-1 ].units.length ){
                throw Error(`The sample_desired_value=[${sample_desired_value}] has ${sample_desired_value.length} elements, But must be ${context.layers[ context.layers.length-1 ].units.length}(the number of units in output layer)`);
            }

            //Do backpropagation and Gradient Descent
            let sample_gradients_data = context.backpropagate_sample(sample_features, sample_desired_value);
        
            let sample_gradients_for_weights = sample_gradients_data['gradients_for_each_weights'];
            let sample_gradients_for_bias    = sample_gradients_data['gradients_of_units'];

            //Accumulate the gradients
            let layersIds = Object.keys(sample_gradients_for_weights);

            layersIds.forEach(function(layerId){
                let layerData  = sample_gradients_for_weights[ layerId ];
                let unitsId    = Object.keys(layerData);

                //If not existis the layer in summed_gradients_for_weights, create with empty object
                if( summed_gradients_for_weights[ layerId ] == undefined ){
                    summed_gradients_for_weights[ layerId ] = {};
                }

                if( summed_gradients_for_bias[ layerId ] == undefined ){
                    summed_gradients_for_bias[ layerId ] = {};
                }

                unitsId.forEach(function(unitId){
                    let number_of_weights = sample_gradients_for_weights[ layerId ][ unitId ].length;

                    //If not exists the unit in summed_gradients_for_weights, create with zeros
                    if( summed_gradients_for_weights[ layerId ][ unitId ] == undefined ){
                        summed_gradients_for_weights[ layerId ][ unitId ] = Array(number_of_weights).fill(0);
                    }

                    //If not exists the unit in summed_gradients_for_bias, create with zero
                    if( summed_gradients_for_bias[ layerId ][ unitId ] == undefined ){
                        summed_gradients_for_bias[ layerId ][ unitId ] = 0;
                    }

                    /**
                    * I do the accumulation in the following way: 
                    * I sum all the gradient of all the weights( of each unit of each layer ), 
                    * in its corresponding position in the hashmap, that is, sample_gradients_for_weights[ layer<LAYER_INDEX> ] [ unit<UNIT_INDEX> ] [ <WEIGHT_INDEX> ] 
                    *
                    * Because this, The format of the output of this sum will be the same format of the "sample_gradients_for_weights" returned by the backpropagate_sample metheod
                    */
                    for( let c = 0 ; c < number_of_weights ; c++ )
                    {   
                        let weight_index = c;
                        let gradient_of_weight = sample_gradients_for_weights[ layerId ][ unitId ][ weight_index ];
                        summed_gradients_for_weights[ layerId ][ unitId ][ weight_index ] = summed_gradients_for_weights[ layerId ][ unitId ][ weight_index ] + gradient_of_weight;
                    }

                    //Sum gradient for acculate the bias(that no have inputs)
                    let gradient_of_bias = sample_gradients_for_bias[ layerId ][ unitId ];
                    summed_gradients_for_bias[ layerId ][ unitId ] = summed_gradients_for_bias[ layerId ][ unitId ] + gradient_of_bias;

                });
            });

        });

        /** BELOW: Do the mean of the gradients of each weight(of each unit of each layer) **/

        /**
        * Struct of the mean_gradients_for_weights:
        * 
        *    mean_gradients_for_weights[ layer<LAYER_INDEX> ][ unit<UNIT_INDEX> ][ <WEIGHT_INDEX> ] = Number
        * 
        *    Or more visual explaination:
        * 
        *    mean_gradients_for_weights
        *     
        *       layer0
        *          --> unit0
        *              --> mean_of_accumulation_of_weight 1 
        *              --> mean_of_accumulation_of_weight 2
        *                  (etc.. other weights gradients) 
        * 
        *              (etc... other units)
        * 
        *      (etc... other layers)
        * 
        *            
        * So, the variable mean_gradients_for_weights Is a hashmap of the mean of the accumulated gradients for each weight( of each unit of each layer ) 
        * Is organized in this way!
        */
        let mean_gradients_for_weights = {}; //TODO RENOMEAR ISSO PRA mean_gradients_for_weights
        
        Object.keys(summed_gradients_for_weights).forEach(function( layerId ){
            let layerData  = summed_gradients_for_weights[ layerId ];
            let unitsId    = Object.keys(layerData);

            //If not existis the layer, create with empty object
            if( mean_gradients_for_weights[ layerId ] == undefined ){
                mean_gradients_for_weights[ layerId ] = {};
            }

            unitsId.forEach(function(unitId){
                //If not existis the unitID, create with empty object
                if( mean_gradients_for_weights[ layerId ][ unitId ] == undefined ){
                    mean_gradients_for_weights[ layerId ][ unitId ] = [];
                }

                //Sum the unit gradient
                let unit_summed_gradient_for_weights = summed_gradients_for_weights[ layerId ][ unitId ];

                let number_of_weights = unit_summed_gradient_for_weights.length;

                for( let c = 0 ; c < number_of_weights ; c++ )
                {
                    let weight_index = c;
                    let sum_of_weight_c = unit_summed_gradient_for_weights[ weight_index ]; //Obtem o peso C que foi acumulado
                    mean_gradients_for_weights[ layerId ][ unitId ][ weight_index ] = sum_of_weight_c / number_of_samples;
                }
            });
        });

        /**
        * Much similar to mean_gradients_for_weights
        * BUT, instead have a sub-object inside the unit object, they have just a value, that is the mean of the accumulated bias(a Number)
        *
        * Or more visual explaination:
        * 
        *    mean_gradients_for_bias
        *     
        *       layer0
        *          --> mean_of_accumulation_of_the_bias_of_unit0
        *          --> mean_of_accumulation_of_the_bias_of_unit1  
        *          --> mean_of_accumulation_of_the_bias_of_unit2        
        *              (etc... other bias means)
        * 
        *       (etc... other layers)
        * 
        * So, inside the mean_gradients_for_bias, he have layers, like in the other examples above
        * And each layer have N means of the accumulation of the bias of each unit( of each layer )
        */
        let mean_gradients_for_bias = {};
        
        Object.keys(summed_gradients_for_bias).forEach(function( layerId ){
            let layerData  = summed_gradients_for_bias[ layerId ];
            let unitsId    = Object.keys(layerData);

            //If not existis the layer, create with empty object
            if( mean_gradients_for_bias[ layerId ] == undefined ){
                mean_gradients_for_bias[ layerId ] = {};
            }

            unitsId.forEach(function(unitId){
                //If not existis the unitID, create with empty zero
                if( mean_gradients_for_bias[ layerId ][ unitId ] == undefined ){
                    mean_gradients_for_bias[ layerId ][ unitId ] = 0;
                }

                //Sum the unit gradient
                let unit_summed_gradient_for_the_bias = summed_gradients_for_bias[ layerId ][ unitId ];
                mean_gradients_for_bias[ layerId ][ unitId ] = unit_summed_gradient_for_the_bias / number_of_samples;
            });
        });

        /**
        * Applies the Gradient Descent algorithm to update the parameters
        */
        context.optimize_the_parameters( mean_gradients_for_weights, mean_gradients_for_bias );
    }

    /**
    * Full Batch training
    * Update the weights just one time per epoch. 
    * 
    * @param {Array} train_samples 
    * @param {Array} number_of_epochs 
    * @returns {Object}
    */
    context.fullbatch_train = function( train_samples, 
                                        number_of_epochs 
    ){
        let currentEpoch    = 0;
        let last_total_loss = 0;
        let loss_history    = [];

        //While the current epoch number is less the number_of_epochs
        while( currentEpoch < number_of_epochs )
        {
            let total_loss = 0;

            //Accumulate the batch and update the parameters
            context.train_fullbatch( train_samples );

            total_loss += context.compute_train_cost( train_samples );

            last_total_loss = total_loss;
            loss_history.push(total_loss);
            
            if( String( Number(currentEpoch / 100) ).indexOf('.') != -1 ){
                console.log(`LOSS: ${last_total_loss}, epoch ${currentEpoch}`)
            }

            //Goto next epoch
            currentEpoch++;
        }

        return {
            last_total_loss: last_total_loss,
            loss_history: loss_history
        };
    }

    /**
    * Mini Batch training
    * Update the weights after every batch. 
    * 
    * @param {Array} train_samples 
    * @param {Array} number_of_epochs 
    * @returns {Object}
    */
    context.minibatch_train = function( train_samples, 
                                        number_of_epochs, 
                                        samples_per_batch=2 
    ){
        let currentEpoch    = 0;
        let last_total_loss = 0;
        let loss_history    = [];

        //Split into N mini batches
        let sub_divisions = [];
        let current_set   = [];

        train_samples.forEach(function( sample_obj, 
                                        sample_index
        ){

            if( current_set.length < samples_per_batch )
            {
                current_set.push( sample_obj );

            //If the current set reach the samples_per_batch limit
            }else{
                sub_divisions.push( [... current_set.copyWithin()] );
                current_set = [];
            }

        });

        //While the current epoch number is less the number_of_epochs
        while( currentEpoch < number_of_epochs )
        {
            let total_loss = 0;
            
            //For each division
            sub_divisions.forEach(function( actual_train_set ){

                /**
                * Accumulate the gradients of each weight of each unit of each layer
                * Then, update the weights and bias in the final of the batch 
                */
                context.train_fullbatch( actual_train_set );

            });

            total_loss += context.compute_train_cost( train_samples );

            last_total_loss = total_loss;
            loss_history.push(total_loss);

            if( String( Number(currentEpoch / 100) ).indexOf('.') != -1 ){
                console.log(`LOSS: ${last_total_loss}, epoch ${currentEpoch}`)
            }

            //Goto next epoch
            currentEpoch++;
        }

        return {
            last_total_loss: last_total_loss,
            loss_history: loss_history
        };
    }

    /**
    * Model training loop 
    * 
    * @param {Array} train_samples
    * @param {Number} number_of_epochs
    */
    context.train = function( train_samples, 
                              number_of_epochs 
    ){

        //Validations
        //If is a array or a instance of net.data.Dataset, is acceptable and compative type
        if( !(train_samples instanceof Array || train_samples.internalType == 'dataset' ) ){
            throw Error(`The train_samples=[${train_samples}] need be a Array instance!`);
        }

        if( train_samples.length == 0 ){
            throw Error(`The train_samples is a empty Array!`);
        }

        if( number_of_epochs == undefined || number_of_epochs == null ){
            throw Error(`The number_of_epochs is undefined!`);
        }

        if( isNaN(number_of_epochs) ){
            throw Error(`The number_of_epochs is NaN!`);
        }

        if( number_of_epochs == Infinity ){
            throw Error(`The number_of_epochs never be inifinity!`);
        }

        if( String(number_of_epochs).indexOf('.') == true ){
            throw Error(`The number_of_epochs=${number_of_epochs} is a invalid value!. The value ${number_of_epochs} should be Integer!`);
        }

        //Task validation
        if( context.task == 'classification' || context.task == 'logistic_regression' || context.task == 'binary_classification'){
            
            //Check if some disired value of the samples ARE NOT BINARY
            let someDesiredValueIsNotBinary = [... train_samples.copyWithin()].some( function(entry) {
                return Object.values(entry[1]).every(function(value) {
                    return (value != 1 && value != 0 && typeof value != 'boolean')
                })
            } )

            if( someDesiredValueIsNotBinary == true ){
                throw Error(`dataset problem!. Some desired values are not binar!. But, In task of ${context.task}, Should be only 0 and 1, or boolean`);
            }

        }

        //Sample validation
        let secure_copy_of_samples_for_validations = [... train_samples.copyWithin()];
        validations.throwErrorIfSomeSampleHaveObjectsArraysInsteadValues( secure_copy_of_samples_for_validations );
        validations.throwErrorIfSomeSampleAreStringsOrCharacters( secure_copy_of_samples_for_validations );
        validations.throwErrorIfSomeSampleAreIncorrectArrayLength( secure_copy_of_samples_for_validations );
        validations.throwErrorIfSomeSampleAreDiffentLengthOfInputsThatTheInputLayer( context.input_layer.inputs , secure_copy_of_samples_for_validations );

        //Start train
        let training_result = {};

        switch( context.getTrainingType() ){
            case 'online':
                training_result = context.online_train(train_samples, number_of_epochs);
                break;

            case 'batch':
            case 'fullbatch':
                training_result = context.fullbatch_train(train_samples, number_of_epochs);
                break;

            case 'minibatch':
                training_result = context.minibatch_train(train_samples, number_of_epochs);
                break;

            default:
                throw Error(`Invalid training type ${ context.getTrainingType() }!`);
        }

        return {
            model: context,
            last_total_loss: training_result.last_total_loss,
            loss_history: training_result.loss_history,
            initial_loss: training_result.loss_history[0],
            final_loss: training_result.loss_history[training_result.loss_history.length-1],
        };

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

    //store the initial weights and biases
    context.initial_weights = context.export();

    return context;
}