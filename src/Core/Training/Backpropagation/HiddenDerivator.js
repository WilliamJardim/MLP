/**
* Calculate a derivative of a especific unit in a hidden layer
* 
* @param {Number} index_of_current_hidden_layer      - The index of the current hidden layer( Who owns the UH unit(that we are calculating the derivative) )
* @param {Number} current_hidden_unit_index          - The index of the UH unit(that we are calculating the derivative)
* @param {Number} weights_of_current_hidden_unit     - The weights of the UH unit(that we are calculating the derivative)
* @param {Array}  current_unit_inputs_values         - The inputs of of the UH unit(that we are calculating the derivative)
* @param {String} current_unit_function_name         - The function name of the UH unit(that we are calculating the derivative)
* @param {Number} current_unit_estimative_value      - The estimated value of the UH unit(that we are calculating the derivative)
*
* @param {Number} next_layer_index                   - The index of the next layer( That is, the layer in front of the current hidden layer )
* @param {Number} number_of_next_layer_units         - The total number of the units of the next layer
* @param {Object} next_layer_units_gradients         - The gradients of the all units in the next layer
*
* @param {Object} map_to_store_gradients - The list to store the calculated gradients with respect each weight and bias of the UH unit(that we are calculating the derivative)
* 
* @returns {Number} - the derivative of the unit
*/
net.MLP.prototype.HiddenLayerDerivator = function(
                                 index_of_current_hidden_layer=Number(), 
                                 current_hidden_unit_index=Number(), 
                                 weights_of_current_hidden_unit=Array(),
                                 current_unit_inputs_values=Array(), 
                                 current_unit_function_name=String(), 
                                 current_unit_estimative_value=Number(), 
                                 
                                 next_layer_index=Number(),
                                 number_of_next_layer_units=Number(),
                                 next_layer_units_gradients={}, 

                                 //List to store the values
                                 map_to_store_gradients={}
){
    let context = this; //The model context

    let derivation_context = {}; //The sub private context to be used to store the values that will used in derivation

    /**
    * Declare the derivative function
    */
    derivation_context.do_derivative = function(){

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
        *                                   (... etc)
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
        *   "next_layer_unit_index" is the number of the current unit in the next layer
        */
        let next_layer_unit_index = 0;
        while( next_layer_unit_index < number_of_next_layer_units ){

            let current_next_unit_index     = next_layer_unit_index;

            let connection_weight_with_UH   = context.getWeightOf({ theWeight: current_hidden_unit_index, 
                                                                    ofUnit: current_next_unit_index, 
                                                                    ofLayer: next_layer_index });

            /**
            * Here I am using "getDerivativeOfBias" to obtaining the derivative of the bias because the derivative of the bias has no entry, 
            * (Why are we obtaining the derivative with respect to the "this hidden unit estimative function", and not with respect to the parameters on which "this hidden unit estimative function" depends.)
            * And what we want is precisely the derivative with respect to the result of the this hidden unit estimation function.  
            * 
            * In other words, here we are obtaining the derivative of the "neural network's estimation function result" with respect to the result of the "current unit function result"
            *
            * To facilitate understanding, you can think that the derivative of the "neural network's estimation function" with respect to the bias of a unit in a hidden layer IS THE SAME THING as the master derivative of that unit, so to speak.
            *
            * So every time you see me referring to "the derivative with respect to the bias of the Unit", you already know that this is the derivative of the "neural network estimative" with respect to the "unit estimative" of the layer.
            */
            let derivative_of_unit = next_layer_units_gradients.getGradientOfAUnit( next_layer_unit_index )
                                                               .getDerivativeOfBias();

            /*
            * NOTE: By using the function: "context.getWeightOf({ theWeight: current_hidden_unit_index, 
            *                                                     ofUnit: current_next_unit_index, 
            *                                                     ofLayer: next_layer_index });", in line 94
            *
            *   The "context.getWeightOf" get the connection weight, whose index is UH(of the external loop in the explanation of the equation above)
            *   Because, for example, if we are calculating the gradient of the first unit in the last hidden layer, 
            *   These gradient(of the hidden unit of the last hidden layer) will depedent of the all gradients in the final layer, 
            *   together with the connection weight, that is, the weight of unit N of the final layer with respect to the hidden unit number UH
            *
            * Above are the gradient equation for the hidden layer units, that are applied in the line below:
            */

            current_unit_accumulator.accumulate( 
                eloh_param  = connection_weight_with_UH,
                LOSS        = derivative_of_unit 
            );

            //Next unit in the next layer
            next_layer_unit_index++;
        };

        let acumulated     = current_unit_accumulator.getAccumulatedValue();

        let derivative     = unit_function_object.derivative( current_unit_estimative_value ) * acumulated;

        /**
        * Compute and store the gradients of this unit
        */
        map_to_store_gradients.storeGradientOf({
            //Unit data
            derivative,
            ofUnit  : current_hidden_unit_index, 

            //Layer data
            ofLayer : index_of_current_hidden_layer
        });

        //Return the actual gradients
        return map_to_store_gradients;
    }
    
    /**
    * The HiddenLayerDerivator it self
    * @returns {Object}
    */
    derivation_context.atSelf = function(){
        return derivation_context;
    }

    /**
    * Returns the derivator object, that will be used in backpropagation 
    */
    return derivation_context;
}