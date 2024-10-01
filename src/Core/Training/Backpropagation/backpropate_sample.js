/**
* Do the backpropagation step for ONE SAMPLE 
* This implementation is entirely original, written from scratch in JavaScript.
*
* This implementation was inspired by various publicly available resources, including concepts 
* and explanations from the work of Jason Brownlee on backpropagation.
* 
* CREDITS && REFERENCE:
* Jason Brownlee, How to Code a Neural Network with Backpropagation In Python (from scratch), Machine Learning Mastery, Available from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/, accessed April 15th, 2024.
* 
*
* @param {Array} sample_inputs  - the sample features
* @param {Array} desiredValuess - the DESIRED values of the final layer units
* 
* @returns {Object} - The mapped gradients of the units of each layer AND The mapped gradients of the each weight of each unit of each layer
*/
net.MLP.prototype.backpropagate_sample = function( sample_inputs  = [], 
                                                   desiredValuess = [] 
){
    //Validations
    if( !(sample_inputs instanceof Array) ){
        throw Error(`The sample_inputs=[${sample_inputs}] need be a Array instance!`);
    }

    if( sample_inputs.length == 0 ){
        throw Error(`The sample_inputs is empty Array!`);
    }

    if( !(desiredValuess instanceof Array) ){
        throw Error(`The desiredValuess=[${desiredValuess}] need be a Array instance!`);
    }

    if( desiredValuess.length == 0 ){
        throw Error(`The desiredValuess is empty Array!`);
    }

    let modelContext                 = this; //The model context
    let number_of_final_layer_units  = modelContext.last_layer.units;
    let final_layer_index            = modelContext.last_layer_index - 1;
    let model_estimatives_data       = modelContext.estimate_values( sample_inputs ); 
    let model_estimatives            = model_estimatives_data.getEstimatedValues();        //Get propagation values
    let estimatives_of_each_layer    = model_estimatives_data.getEstimativesOfEachLayer(); //Get the estimative-values of each unit of each layer
    let inputs_of_each_layer         = model_estimatives_data.getInputsOfEachLayer();

    /**
    * Store the gradients of the weights and bias, each unit of each layer 
    * 
    * "gradients_per_layer" structure: 
    * layer
    *    --> unit == GradientVector
    * 
    * That is, the "gradients_per_layer" contains "layers" as keys, and the "layers" contains "units" as keys, and the "units" is GradientVector instances
    */
    let gradients_per_layer = {};

    /**
    * Calculate the derivative of each unit in final layer
    * This process is made by a subtraction of the "unit estimated value" and the "desired value for the unit".
    * So, Each unit have a "desired value", and each unit produces a "estimative" in the "estimate_values" phase, so these informations are used to calculate this derivatives
    * 
    * And these gradients will be stored in the "gradients_per_layer"
    *
    * So to do this, i will use the following below:
    */
    gradients_per_layer[ `layer${ final_layer_index }` ] = {};

    for( let unit_number = 0 ; unit_number < number_of_final_layer_units ; unit_number++  )
    {
        let unit_obj           = modelContext.getLayer(final_layer_index)
                                             .getUnit( unit_number );

        let unit_inputs        = inputs_of_each_layer[ `layer${ final_layer_index }` ];

        let unit_function      = net.activations[ unit_obj.activation_function ];
        let estimative         = model_estimatives[ unit_number ];
        let desired_value      = desiredValuess[ unit_number ];
        
        let loss_wrt_unit_estimation = ( estimative - desired_value ) * unit_function.derivative( estimative );

        /**
        * Compute the partial derivatives of each weight parameter( that is the gradient vector ), and store as GradientVector instance
        */
        gradients_per_layer[ `layer${ final_layer_index }` ][ `unit${ unit_number }` ] = net.GradientVector({

                                                                    //Repass the LOSS WRT OF THE UNIT ESTIMATION FUNCTION
                                                                    loss_wrt_unit_estimation, 

                                                                    //Get the inputs of the weights of the current unit
                                                                    unit_inputs

                                                                  });
    }

    /** 
    * Start the backpropagation
    * Starting in LAST HIDDEN LAYER and going in direction of the FIRST HIDDEN LAYER.
    * 
    * When we reach the FIRST HIDDEN LAYER, when had calculate the gradients of all units in the FIRST HIDDEN LAYER, the backpropagation finally ends.
    */

    let number_of_layers = modelContext.number_of_layers - 1 - 1 - 1; //Ignore the input layer and ignore the final layer

    /**
    * While the "while loop" not arrived the first hidden layer
    * The first hidden layer(that have index 0, will be the final layer that will be computed) 
    */
    let current_hidden_layer = number_of_layers;
    while( current_hidden_layer >= 0 )
    {
        gradients_per_layer[ `layer${ current_hidden_layer }` ] = {};

        /*
        * THE FORMULA USED IS FOLLOWING:
        *
        * The gradients for the units in ANY hidden layer is:
        * Below are a simple example suposing that in the next layer we have 2 units:
        * 
        * >>> EQUATION WITH A EXAMPLE OF USE:
        * 
        *    current_unit<UH>_derivative  = ( derivative_of_next_layer_unit<N0> * weight<UH>_of_unit<N0> ) + 
        *                                   ( derivative_of_next_layer_unit<N1> * weight<UH>_of_unit<N1> ) + 
        *                                   ( ... etc )
        * 
        *    NOTE: In this example, the next layer have just 2 units(N0 and N1, respectively), 
        *          but There could be as many as there were. By this, i put "[... etc]", to make it clear that there could be more than just 2 units
        * 
        *    NOTE: "current_unit" is a hidden unit!
        *          So the "current_unit<UH>" is the hidden unit of index <UH> in current hidden layer.
        *          Then, the "current_unit<UH>_derivative" is the derivative of the LOSS with respect to the unit "current_unit<UH>".
        * 
        * 
        * >>> EXPLANATION:
        * 
        *   Where the UH is the index of the hidden unit(which we are calculating the derivative) in the current hidden layer. And the Ns( N0, N1, etc... ) are the indexes of the next layer units.
        *   Relemering that the in the example above, we have just 2 units in the next layer, so the have only the N0(unit one) and N1(unit two).
        * 
        *   The weight<UH> is the weight that make a eloh( unites the two parts, that is, which connects the estimative-value of unit H with the input of unit U), of weights array in the next_layer_unit<N> object.
        * 
        * 
        * This is the equation that are used for apply the backpropagation. This equation is used in this loop.
        * So the code bellow apply this:
        * 
        */

        /**
        * For each hidden unit in the current hidden layer
        * (The logic for any hidden layer unit will always be the same, for any unit of any hidden layer)
        */
        for( let hidden_unit_number = 0 ; hidden_unit_number < modelContext.getLayer(current_hidden_layer).getUnits().length ; hidden_unit_number++  )
        {
            let hidden_unit_obj           = modelContext.getLayer( current_hidden_layer )
                                                        .getUnit( hidden_unit_number );

            /**
            * I make a copy of the "inputs_of_each_layer[ `layer${ current_hidden_layer }` ]" variable to make a safe and independent copy of the same layer inputs for all units,
            * 
            * Because, the key point of this is: 
            * 
            *    All units receives the same inputs!. That is, the same inputs of the layer Who owns the unit
            *    Because, each layer have N inputs. And the inputs of EACH UNIT in a layer are the estimated values of the previous layer.
            *    So, in short, in a given layer of the neural network, all units in that layer will receive exactly the same inputs, THAT IS, THE ESTIMATED VALUE OF THE PREVIOUS LAYER, WHICH ARE ITS INPUTS   
            *
            *    Except the input layer, because the input layer has no units, It only has inputs(just numbers), and nothing more than that.
            *    Therefore, the input layer has no units, and therefore does not receive input from a previous layer
            *
            *    To facilitate and standardize the process, in a generic way for each layer, we can imagine that the estimated values of the previous layer are the inputs themselves.               
            */
            let unit_inputs               = inputs_of_each_layer[ `layer${ current_hidden_layer }` ].copyWithin();

            let hidden_unit_function      = net.activations[ hidden_unit_obj.activation_function ];
            let hidden_unit_estimative    = estimatives_of_each_layer[ `layer${ current_hidden_layer }` ][ `unit${ hidden_unit_number }` ];

            let loss_wrt_unit_estimation = 0;

            /**
            * GET THE NEXT LAYER:
            */
            modelContext.getLayer( current_hidden_layer )
                        .getNextLayer( ( next_layer_context, next_layer_index ) => {
                             
                             /**
                             * For each unit in LEXT LAYER 
                             * We are working in the context of the next layer:
                             * 
                             *   "layer_index" is the number of the current unit in the next layer
                             */
                             ( next_layer_context )
                             .getUnits().forEach( ( unit_obj, unit_index ) => {
                                    const layer_index      = next_layer_index;
                                    const layer_gradients  = gradients_per_layer[ `layer${ layer_index }` ];
                                    const unit_derivative  = layer_gradients[ `unit${ unit_index }` ].get_wrt_unit_estimation();

                                    /**
                                    * NOTE: By using the function: "modelContext.getWeightOf({
                                    *                                    theWeight : hidden_unit_number,
                                    *                                    ofUnit    : unit_index,
                                    *                                    ofLayer   : unit_obj.getLayerOfThisUnit().getIndex() //Get the layer of the next unit
                                    *                                });", below, in line 196
                                    *
                                    *   The "context.getWeightOf" returns the parameter of the "current unit in the next layer, A.K.A called N in the formula above" 
                                    *   that make the eloh with the "current unit in the current hidden layer, AKA UH unit in the formula above", Whose index is UH(of the external loop in the explanation of the equation above)
                                    *
                                    *   Because, for example, if we are calculating the gradient of the first unit in the last hidden layer, 
                                    *   These gradient(of the hidden unit of the last hidden layer) will depedent of the all gradients in the final layer, 
                                    *   together with the eloh parameter, that is, the weight of unit N of the final layer with respect to the hidden unit number UH
                                    *
                                    * Above are the gradient equation for the hidden layer units, that are applied in the line below:
                                    */
                                    const eloh_parameter   = modelContext.getWeightOf({
                                                                theWeight : hidden_unit_number,
                                                                ofUnit    : unit_index,
                                                                ofLayer   : unit_obj.getLayerOfThisUnit().getIndex() //Get the layer of the next unit
                                                             });

                                    loss_wrt_unit_estimation += ( unit_derivative * 
                                                                  eloh_parameter * 
                                                                  hidden_unit_function.derivative( hidden_unit_estimative ) );

                                    //Alert if something is wrong
                                    if( isNaN(unit_derivative) || isNaN(unit_derivative) ){ debugger; }

                                    
                             });

                        });
                        
            //Compute the partial derivatives of each weight parameter( that is the gradient vector ), and store as GradientVector instance
            gradients_per_layer[ `layer${ current_hidden_layer }` ][ `unit${hidden_unit_number}` ] = net.GradientVector({

                                                                        //Repass the LOSS WRT OF THE UNIT ESTIMATION FUNCTION
                                                                        loss_wrt_unit_estimation,

                                                                        //Get the inputs of the weights of the current unit
                                                                        unit_inputs
                                                                    });
        }

        /** Previous layer **/
        current_hidden_layer--;

    }

    /**
    * Split the gradients in two distinct objects 
    */
    let gradients_of_each_unit_bias_per_layer = {};
    let gradients_of_each_unit_weights_per_layer = {};
    
    let layersKeys = Object.keys( gradients_per_layer );
    layersKeys.forEach((layerId)=>{
        let unitsKeys = Object.keys( gradients_per_layer[layerId] );
        gradients_of_each_unit_bias_per_layer[ layerId ] = {};
        gradients_of_each_unit_weights_per_layer[ layerId ] = {};

        unitsKeys.forEach((unitId)=>{
            gradients_of_each_unit_bias_per_layer[ layerId ][ unitId ]  = gradients_per_layer[layerId][unitId].loss_wrt_unit_estimation;
            gradients_of_each_unit_weights_per_layer[ layerId ][ unitId ] = gradients_per_layer[layerId][unitId].loss_wrt_unit_weights;
        });
    });

    /**
    * Return the gradients
    */
    return {
        gradients_per_layer,
        gradients_of_each_unit_weights_per_layer,
        gradients_of_each_unit_bias_per_layer
    };
    
}