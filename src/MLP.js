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

        let output = net.activations[context.activation_function](sum);
        return output;
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

    context.hyperparameters    = config_dict['hyperparameters'];
    context.learning_rate      = context.hyperparameters['learningRate'];

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
        * So, the property LAYER_INPUTS of the first hidden layer is the sample_inputs. And in the feedforward, each layer(L) will have the property LAYER_INPUTS, storing the outputs of the previous layer(L-1) 
        * 
        * So, the inputs of first hidden layer( that is L=0 ), will be the sample_inputs
        * And the inputs of secound hidden layer( that is L=1 ), will be the outputs of the first hidden layer( that is L=0 )
        *
        * always in this way.
        */
        context.layers[0]['LAYER_INPUTS'] = [... sample_inputs];

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

            //For each unit in current layer L, get the UNIT OUTPUT and store inside the unit
            let units_outputs = [];
            for( let U = 0 ; U < number_of_units ; U++ )
            {
                let current_unit = units_in_layer[ U ];

                let unit_output  = current_unit.estimateOutput( current_layer['LAYER_INPUTS'] );
                current_unit['ACTIVATION'] = unit_output;
                //The inputs is the same of all units in a layer
                current_unit['INPUTS'] = current_layer['LAYER_INPUTS'];

                units_outputs.push( unit_output );
            }

            /*
            * The inputs of a layer L is always the outputs of previous layer( L-1 ) 
            * Then the in lives below will Store the outputs of the current layer(L) in the NEXT LAYER(L+1) AS INPUTS
            */

            //If the current layer(L) is NOT the output layer
            if( current_layer.layer_type != 'output' ){
                let next_layer = context.layers[ L+1 ];
                
                //Set the current layer(L) outputs AS INPUTS OF THE NEXT LAYER(L+1)
                next_layer['LAYER_INPUTS'] = units_outputs;
            }

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
        let output_estimated_values  = context.feedforward_sample( sample_inputs );
        let number_of_output_units   = output_estimated_values.length;
        let layers                   = context.layers;
        let number_of_layers         = layers.length;

        //Calculate the LOSS of each output unit and store in the outputs units
        for( let U = 0 ; U < number_of_output_units ; U++ )
        {
            let output_unit            = context.layers[ context.layers.length-1 ].units[ U ];
            let unitActivationFn       = output_unit['activation_function'];
            
            let unitOutput             = output_estimated_values[ U ];
            
            let desiredOutput     = desiredOutputs[ U ];
            let unitError         = unitOutput - desiredOutput;
            
            //The derivative of activation funcion of the U output(at output layer)
            let outputDerivative  = net.activations[ unitActivationFn ].derivative( unitOutput );

            //The delta of this output unit U
            let unit_nabla = unitError * outputDerivative;

            //Store the error in the unit
            output_unit['LOSS'] = unit_nabla;
        }

        //Start the backpropagation
        //A reverse for(starting in OUTPUT LAYER and going in direction of the FIRST HIDDEN LAYER)
        for( let L = number_of_layers-1-1; L >= 0 ; L-- )
        {
            //Current layer data
            let current_layer                  = context.layers[ L ];
            let current_layer_units            = current_layer.units;
            let number_of_units_current_layer  = current_layer_units.length;

            //Next layer data
            let next_layer                     = context.layers[ L+1 ];
            let number_of_next_layer_units     = next_layer.units.length;

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

                let current_hidden_layer_unit = current_layer_units[ UH ]; //The hidden layer unit of number UH(like in the equation above)
            
                // Do the sum of the errors
                let current_hidden_unit_LOSS = 0;

                /** For each unit N in LEXT LAYER( L+1 ) **/
                for( let N = 0 ; N < number_of_next_layer_units ; N++ )
                {
                    let next_layer_unit_N          = next_layer.units[ N ];
                    let connection_weight_with_UH  = next_layer_unit_N.weights[ UH ];
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
                let unit_nabla = current_hidden_unit_LOSS * net.activations[ current_hidden_layer_unit.activation_function ].derivative( current_hidden_layer_unit.ACTIVATION );
                current_hidden_layer_unit.LOSS = unit_nabla;
            }
        }

        //Update weights and Bias using Gradient Descent
        for( let L = 0 ; L < number_of_layers ; L++ )
        {
            let current_layer = context.layers[ L ];
            let current_layer_units = current_layer.units;
            let number_of_units_current_layer = current_layer_units.length;

            //For each unit in current layer
            for( let U = 0 ; U < number_of_units_current_layer ; U++ )
            {
                let current_unit = current_layer_units[ U ];

                //For each weight
                for( let W = 0 ; W < current_unit.weights.length ; W++ )
                {
                    current_unit.weights[ W ] -= context.learning_rate * current_unit['LOSS'] * current_unit['INPUTS'][ W ];
                }

                //Update bias
                current_unit.bias -= context.learning_rate * current_unit['LOSS'];

            }

        }
    }

    /**
    * Compute de COST
    */
    context.compute_train_cost = function( train_samples ){
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

    return context;
}