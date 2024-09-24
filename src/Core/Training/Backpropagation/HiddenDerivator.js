/**
* Calculate a derivative of a especific unit in a hidden layer
* 
* @param {Number} index_of_current_hidden_layer      - The index of the current hidden layer( Who owns the UH unit(that we are calculating the derivative) )
* @param {Number} current_hidden_unit_index          - The index of the UH unit(that we are calculating the derivative)
* @param {Number} weights_of_current_hidden_unit     - The weights of the UH unit(that we are calculating the derivative)
* @param {Array}  current_unit_inputs_values         - The inputs of of the UH unit(that we are calculating the derivative)
* @param {String} current_unit_function_name         - The function name of the UH unit(that we are calculating the derivative)
* @param {Number} current_unit_estimative_value          - The estimated value of the UH unit(that we are calculating the derivative)
* @param {Array}  next_layer_units_objects           - The units of the next layer
* @param {Object} next_layer_units_gradients         - The gradients of the all units in the next layer
*
* @param {Object} map_to_store_gradients_of_units    - The list to store the calculated gradients of the UH unit(that we are calculating the derivative)
* @param {Object} map_to_store_gradients_for_weights - The list to store the calculated gradients with respect each weight of the UH unit(that we are calculating the derivative)
* 
* @returns {Number} - the derivative of the unit
*/
net.HiddenLayerDerivator = function(
                                 index_of_current_hidden_layer=Number(), 
                                 current_hidden_unit_index=Number(), 
                                 weights_of_current_hidden_unit=Array(),
                                 current_unit_inputs_values=Array(), 
                                 current_unit_function_name=String(), 
                                 current_unit_estimative_value=Number(), 
                                 next_layer_units_objects=Array(), 
                                 next_layer_units_gradients={}, 

                                 //List to store the values
                                 map_to_store_gradients_of_units={}, 
                                 map_to_store_gradients_for_weights={}
){
    let context = {};
    context.index_of_current_hidden_layer  = index_of_current_hidden_layer;
    context.current_hidden_unit_index      = current_hidden_unit_index;
    context.weights_of_current_hidden_unit = weights_of_current_hidden_unit;
    context.current_unit_inputs_values     = current_unit_inputs_values;
    context.current_unit_function_name     = current_unit_function_name;
    context.current_unit_estimative_value      = current_unit_estimative_value;
    context.next_layer_units_objects       = next_layer_units_objects;
    context.next_layer_units_gradients     = next_layer_units_gradients;

    context.map_to_store_gradients_of_units = map_to_store_gradients_of_units;
    context.map_to_store_gradients_for_weights = map_to_store_gradients_for_weights;

    context.derivate = function(){

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

        let unit_derivative = acumulated * unit_function_object.derivative( current_unit_estimative_value );

        /**
        * Store the gradient in gradients object
        */
        context.map_to_store_gradients_of_units[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ] = unit_derivative;

        /*
        * Aditionally, store the erros TOO with respect of each weight
        */
        context.map_to_store_gradients_for_weights[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ] = [];

        context.weights_of_current_hidden_unit.forEach(function( weight_value, 
                                                                 weight_index_c
        ){

            let weight_input_C = context.current_unit_inputs_values[ weight_index_c ]; //CRIAR UM GETTER  
            context.map_to_store_gradients_for_weights[ `layer${ index_of_current_hidden_layer }` ][ `unit${ current_hidden_unit_index }` ][ weight_index_c ] = unit_derivative * weight_input_C;

        });

        //Return the actual gradients
        return {
            map_to_store_gradients_of_units     : context.map_to_store_gradients_of_units,
            map_to_store_gradients_for_weights  : context.map_to_store_gradients_for_weights
        }
    }
    
    return context;
}