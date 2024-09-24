/**
* Calculate the derivative of each unit in the output layer
* 
* @param {Array}  output_estimated_values              - The network estimations(of each unit)
* @param {Array}  desiredOutputs                       - The desired values(for each unit)
* @param {Object} list_to_store_gradients_of_units     - A object to store the gradients of each unit
* @param {Object} list_to_store_gradients_for_weights  - A object to store the gradients of each unit with respect to each unit weight
*
* @returns {Object} - The calculated gradients of the output layer
* 
*/
net.MLP.prototype.calculate_derivatives_of_output_units = function(output_estimated_values, 
                                                                   desiredOutputs, 
                                                                   list_to_store_gradients_of_units, 
                                                                   list_to_store_gradients_for_weights 
){
    let context = this; //The model context
    
    /**
    * Create a derivator for derivate the output layer 
    */
    let derivatorInstance = net.OutputDerivator(context, 
                                                output_estimated_values, 
                                                desiredOutputs, 
                                                list_to_store_gradients_of_units, 
                                                list_to_store_gradients_for_weights);

    /**
    * Get and return the gradients of the last model layer 
    */
    let outputGradients = derivatorInstance.derivate();
    return outputGradients;
}