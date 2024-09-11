/**
* Multilayer Perceptron Neural Network (MLP)
* By William Alves Jardim
* 
* CREDITS && REFERENCE:
* 
*   Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
* See README.md
*/
var net = {};

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
* A utility for update weights
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

//A Unit(with just feedforward and weight initialization)
net.Unit = function( unit_config={} ){
    let context = {};

    //Parameters
    context.number_of_inputs      = unit_config.number_of_inputs     || Number();
    context.activation_function   = unit_config.activation_function  || 'sigmoid';
    context.weights               = unit_config.weights              || Array(context.number_of_inputs).fill(0);
    context.bias                  = unit_config.bias                 || Number();

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

        let sum = 0;
        for( let i = 0 ; i < number_of_inputs ; i++ ) {
            sum = sum + ( sample_inputs[i] * context.weights[i] );
        }
        //Add the bias
        sum = sum + context.bias;

        let output = net.activations[ context.getFunctionName() ](sum);

        return {
            activation_function_output: output,
            unit_potential: sum //The unit activation potential(just the sum)
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
        return context.INPUTS[ weight_index ];
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

       context.units[i].generate_random_parameters( context.number_of_inputs );
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
    * Set the inputs of this layer, to be used in context.get_unit_outputs
    */
    context.setInputs = function( LAYER_INPUTS=[] ){
        if( !(LAYER_INPUTS instanceof Array) ){
            throw Error(`LAYER_INPUTS need be a Array!`);
        }

        context['LAYER_INPUTS'] = LAYER_INPUTS;
    }

    /**
    * Get the inputs of this layer, to be used in context.get_unit_outputs
    */
    context.getInputs = function(){
        return context['LAYER_INPUTS'];
    }

    /**The unit outputs. */
    context.get_unit_outputs = function(){
        let number_of_units = context.getUnits().length;

        //For each unit in this layer L, get the UNIT OUTPUT and store inside the unit
        let units_outputs = [];

        for( let U = 0 ; U < number_of_units ; U++ )
        {
            let LAYER_INPUTS = context.getInputs();

            let current_unit = context.getUnit( U );

            let unit_output_data  = current_unit.estimateOutput( LAYER_INPUTS );
            let act_potential     = unit_output_data.unit_potential;
            let unit_output       = unit_output_data.activation_function_output;

            current_unit['UNIT_POTENTIAL'] = act_potential;

            current_unit['UNIT_OUTPUT'] = unit_output; //So important in backpropagation and gradient descent steps

            //The inputs is the same of all units in a layer
            current_unit['INPUTS'] = LAYER_INPUTS;

            units_outputs.push( unit_output );
        }

        return units_outputs;
    }

    //Return the layer ready
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
    context.hyperparameters    = config_dict['hyperparameters'];
    context.learning_rate      = context.hyperparameters['learningRate'];

    //Class parameters and model Hyperparameters validations
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

        let number_of_layers      = context.layers.length;

        /**
        * The inputs of a layer L is always the outputs of previous layer( L-1 )
        * So, the property LAYER_INPUTS of the first hidden layer is the sample_inputs. And in the feedforward, each layer(L) will have the property LAYER_INPUTS, storing the outputs of the previous layer(L-1) 
        * 
        * So, the inputs of first hidden layer( that is L=0 ), will be the sample_inputs
        * And the inputs of secound hidden layer( that is L=1 ), will be the outputs of the first hidden layer( that is L=0 )
        *
        * Always in this way.
        */
        context.get_first_hidden_layer().setInputs( [... sample_inputs] );

        //The outputs of OUTPUT LAYER
        let final_outputs         = []; 

        /**
        * In this case, the layer 0 is the first hidden layer, because the input layer is ignored in initialization
        */
        for( let L = 0 ; L < number_of_layers ; L++ )
        {
            let current_layer = context.getLayer( L );
            
            //For each unit in current layer L, get the UNIT OUTPUT and store inside the unit
            let units_outputs = current_layer.get_unit_outputs();
            
            //If the current layer(L) is NOT the output layer
            if( current_layer.notIs('output') ){

                /*
                * The inputs of a layer L is always the outputs of previous layer( L-1 ) 
                * Then the in lines below will Store the outputs of the current layer(L) in the NEXT LAYER(L+1) AS INPUTS
                */
                let next_layer = context.getLayer( L+1 );
                
                //Set the current layer(L) outputs AS INPUTS OF THE NEXT LAYER(L+1)
                next_layer.setInputs( units_outputs );
            }

            //If is the output layer
            if( current_layer.is('output') )
            {
                final_outputs = units_outputs;
            }
        }

        //Return the final outputs( that is the outputs of the output layer )
        return final_outputs;
    }

    /* Get the output layer */
    context.getOutputLayer = function(){
        return context.layers[ context.layers.length-1 ];
    }

    /**
    * Do the backpropagation step for ONE SAMPLE 
    *
    * @param {Array} sample_inputs  - the sample features
    * @param {Array} desiredOutputs - the DESIRED outputs of the output units
    */
    context.backpropagate_sample = function( sample_inputs=[], desiredOutputs=[] ){
        
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

        //Do the feedforward step
        let output_estimated_values  = context.feedforward_sample( sample_inputs );
        let number_of_output_units   = output_estimated_values.length;
        let number_of_layers         = context.getLayers().length;

        //Calculate the LOSS of each output unit and store in the outputs units
        for( let U = 0 ; U < number_of_output_units ; U++ )
        {
            let output_unit            = context.getOutputLayer().getUnit( U );
            let unitActivationFn       = output_unit.getFunctionName();
            
            let unitOutput             = output_estimated_values[ U ];
            
            let desiredOutput     = desiredOutputs[ U ];
            let unitError         = unitOutput - desiredOutput;
            
            //The derivative of activation funcion of the U output(at output layer)
            let outputDerivative  = net.activations[ unitActivationFn ].derivative( unitOutput );

            //The delta of this output unit U
            let unit_nabla = unitError * outputDerivative;

            //Store the error in the unit
            output_unit.LOSS = unit_nabla;
        }

        //Start the backpropagation
        //A reverse for(starting in OUTPUT LAYER and going in direction of the FIRST HIDDEN LAYER)
        for( let L = number_of_layers-1-1; L >= 0 ; L-- )
        {
            //Current layer data
            let current_layer                  = context.getLayer( L );
            let number_of_units_current_layer  = current_layer.getUnits().length;

            //Next layer data
            let next_layer                     = context.getLayer( L+1 );
            let number_of_next_layer_units     = next_layer.getUnits().length;

            //For each unit in CURRENT HIDDEN LAYER
            for( let UH = 0 ; UH < number_of_units_current_layer ; UH++ )
            {
                /*
                * The gradients for the units in ANY hidden layer is:
                * Below are a simple example suposing that in the next layer we have 2 units:
                * 
                * >>> EQUATION WITH A EXAMPLE OF USE:
                * 
                *    current_layer_unit<UH>_error = (next_layer_unit<N0>.weight<UH> * next_layer_unit<N0>.LOSS) + 
                *                                   (next_layer_unit<N1>.weight<UH> * next_layer_unit<N1>.LOSS) + 
                *                                   [... etc]
                * 
                *    NOTE: In this example, the next layer have just 2 units(N0 and N1, respectively), 
                *          but There could be as many as there were. By this, i put "[... etc]", to make it clear that there could be more than just 2 units
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

                let current_hidden_layer_unit = current_layer.getUnit( UH ); //The hidden layer unit of number UH(like in the equation above)
            
                // Do the sum of the errors
                let current_hidden_unit_LOSS = 0;

                /** For each unit N in LEXT LAYER( L+1 ) **/
                for( let N = 0 ; N < number_of_next_layer_units ; N++ )
                {
                    let next_layer_unit_N          = next_layer.getUnit( N );
                    let connection_weight_with_UH  = next_layer_unit_N.getWeight( UH );
                    let LOSS_of_unit_N             = next_layer_unit_N.LOSS;

                    /**
                    * NOTE: The next_layer_unit_N.weights[ UH ] is the connection weight, whose index is UH(of the external loop in the explanation of the equation above)
                    *       Because, for example, if we are calculating the gradient of the first unit in the last hidden layer, 
                    *       These gradient(of the hidden unit) will depedent of the all gradients in the output layer, 
                    *       together with the connection weight, that is, the weight of unit N of the output layer with respect to the hidden unit number UH
                    *
                    * Above are the gradient equation for the hidden layer units, that are applied in the line below:
                    */

                    current_hidden_unit_LOSS += ( connection_weight_with_UH * LOSS_of_unit_N );
                }

                //Store the error in the unit
                let unit_nabla = current_hidden_unit_LOSS * net.activations[ current_hidden_layer_unit.getFunctionName() ].derivative( current_hidden_layer_unit.UNIT_OUTPUT );
                current_hidden_layer_unit.LOSS = unit_nabla;
            }
        }

        //Update weights and Bias using Gradient Descent
        for( let L = 0 ; L < number_of_layers ; L++ )
        {
            let current_layer = context.getLayer( L );
            let number_of_units_current_layer = current_layer.getUnits().length;

            //For each unit in current layer
            for( let U = 0 ; U < number_of_units_current_layer ; U++ )
            {
                let current_unit = current_layer.getUnit( U );

                //For each weight
                for( let W = 0 ; W < current_unit.getWeights().length ; W++ )
                {
                    //current_unit.subtractWeight( W, context.learning_rate * current_unit.LOSS * current_unit.getInputOfWeight( W ) );
                    current_unit.selectWeight( W )
                                .subtract( context.learning_rate * current_unit.LOSS * current_unit.getInputOfWeight( W ) );
                }

                //Update bias
                current_unit.subtractBias( context.learning_rate * current_unit.LOSS );

            }

        }
    }

    /**
    * Compute de COST
    */
    context.compute_train_cost = function( train_samples ){

        //Validations
        if( !(train_samples instanceof Array) ){
            throw Error(`The train_samples=[${train_samples}] need be a Array instance!`);
        }

        if( train_samples.length == 0 ){
            throw Error(`The train_samples is a empty Array!`);
        }

        let cost = 0;
        
        for( let A = 0 ; A < train_samples.length ; A++ )
        {   
            let sample_data            = train_samples[ A ];
            let sample_features        = sample_data[0]; //SAMPLE FEATURES
            let sample_desired_value   = sample_data[1]; //SAMPLE DESIRED OUTPUTS
            let estimatedValues        = context.feedforward_sample(sample_features);

            for( let S = 0 ; S < estimatedValues.length ; S++ )
            {
                cost += ( sample_desired_value[ S ] - estimatedValues[ S ] ) ** 2;
            }

        }

        return cost;
    }

    /**
    * Model training loop 
    * 
    * @param {Array} train_samples
    * @param {Number} number_of_epochs
    */
    context.train = function( train_samples, number_of_epochs ){

        //Validations
        if( !(train_samples instanceof Array) ){
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

        let last_total_loss = 0;
        let loss_history = [];

        //For each epoch
        for( let p = 0 ; p < number_of_epochs ; p++ )
        {
            let total_loss = 0;

            //Training process
            for( let i = 0 ; i < train_samples.length ; i++ )
            {
                let sample_data             = train_samples[i];
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
                context.backpropagate_sample(sample_features, sample_desired_value);
            }

            total_loss += context.compute_train_cost( train_samples );

            last_total_loss = total_loss;
            loss_history.push(total_loss);
            
            if( String( Number(p / 100) ).indexOf('.') != -1 ){
                console.log(`LOSS: ${last_total_loss}, epoch ${p}`)
            }
        }

        return {
            model: context,
            last_total_loss: last_total_loss,
            loss_history: loss_history,
            initial_loss: loss_history[0],
            final_loss: loss_history[loss_history.length-1],
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