/**
* A simple Multilayer Perceptron implementation in JavaScript
* by William Alves Jardim
*/
var net = {};

//Activation functions
net.activations = {

    sigmoid: function(x) {
        let functionOutput =  1.0 / (1.0 + Math.exp(-x));
        let derivative     =  functionOutput * (1.0 - functionOutput)
    
        //Compute the function output and derivative
        return {
            functionValue   : functionOutput,
            derivativeValue : derivative
        }
    }

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
    * Generate random parameters
    */
    context.generate_random_parameters = function( number_of_inputs=context.number_of_inputs ){

        for( let i = 0 ; i < number_of_inputs ; i++ )
        {
            context.weights[i] = Math.random() * 1.1;
        }

        context.bias = Math.random() * 1.1;
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

        let output = net.activations[context.activation_function](sum);
        return output['functionValue'];
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
    * Feedforward for a ONE SAMPLE
    * @param {Array} sample_inputs 
    * @returns {Array}
    */
    context.feedforward_sample = function( sample_inputs=[] ){

        let number_of_layers      = context.layers.length;

        /**
        * The inputs of a layer L is always the outputs of previous layer( L-1 )
        * So, the current_layer_inputs starts being sample_inputs, and after each layer propagation, this variavel will be updated
        * 
        * So, the inputs of first hidden layer( that is L=0 ), will be the sample_inputs
        * And the inputs of secound hidden layer( that is L=1 ), will be the outputs of the first hidden layer( that is L=0 )
        *
        * always in this way.
        */
        let current_layer_inputs  = [... sample_inputs]; 

        //The outputs of OUTPUT LAYER
        let final_outputs         = []; 

        /**
        * In this case, the layer 0 is the first hidden layer, because the input layer is ignored in initialization
        */
        for( let L = 0 ; L < number_of_layers ; L++ )
        {
            let current_layer   = context.layers[ L ];
            let units_in_layer  = current_layer.units;
            let number_of_units = units_in_layer.length;

            //For each unit in current layer L, get the UNIT OUTPUT
            let units_outputs = [];
            for( let U = 0 ; U < number_of_units ; U++ )
            {
                let current_unit = units_in_layer[ U ];
                let unit_output  = current_unit.estimateOutput( current_layer_inputs );
                units_outputs.push( unit_output );
            }

            //The inputs of a layer L is always the outputs of previous layer( L-1 )
            current_layer_inputs = units_outputs;

            //If is the output layer
            if( current_layer.layer_type == 'output' )
            {
                final_outputs = units_outputs;
            }
        }

        //Return the final outputs( that is the outputs of the output layer )
        return final_outputs;
    }

    /**
    * Do the backpropagation step for ONE SAMPLE 
    * 
    * @param {Array} sample_inputs  - the sample features
    * @param {Array} desiredOutputs - the DESIRED outputs of the output units
    */
    context.backpropagate_sample = function( sample_inputs=[], desiredOutputs=[] ){
        //Do the feedforward step
        let output_estimated_values = context.feedforward_sample( sample_inputs );
        let number_of_units         = output_estimated_values.length;

        //Calculate the LOSS of each output unit
        let LOSS_OF_OUTPUT_UNITS = [];
        for( let U = 0 ; U < number_of_units ; U++ )
        {
            let unitOutput     = output_estimated_values[ U ];
            let desiredOutput  = desiredOutputs[ U ];
            let unitError          = unitOutput - desiredOutput;
            LOSS_OF_OUTPUT_UNITS.push( unitError );
        }

        //TODO: calcular os deltas, multiplicando o erro pela derivada


        //Start the backpropagation
        let number_of_layers = model.layers.length;

        //A reverse for(starting in OUTPUT LAYER and going in direction of the FIRST HIDDEN LAYER)
        for( let L = number_of_layers-1; L >= 0 ; L-- )
        {
            let current_layer = model.layers[ L ];

            
        }
    }

    return context;
}