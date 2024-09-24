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
net.MLP.prototype.calculate_hidden_unit_derivative = function(index_of_current_hidden_layer=Number(), 
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
    /**
    * Create a derivator for derivate the last layer 
    */
    let derivatorInstance = net.HiddenLayerDerivator(index_of_current_hidden_layer, 
                                                     current_hidden_unit_index, 
                                                     weights_of_current_hidden_unit,
                                                     current_unit_inputs_values, 
                                                     current_unit_function_name, 
                                                     current_unit_estimative_value, 
                                                     next_layer_units_objects, 
                                                     next_layer_units_gradients, 

                                                     //List to store the values
                                                     map_to_store_gradients_of_units, 
                                                     map_to_store_gradients_for_weights);

    /**
    * Get and return the gradients of a hidden layer 
    */
    let hiddenGradients = derivatorInstance.derivate();
    return hiddenGradients;
}