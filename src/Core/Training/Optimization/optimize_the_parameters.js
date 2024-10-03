/**
* Applies the Gradient Descent algorithm 
* Is used to update the weights and bias of each unit in each layer
* 
* @param {Object} the_gradients_for_weights - The gradients of each weight of each unit of each layer
* @param {Object} the_gradients_for_bias    - The gradients of the bias of each unit of each layer
*/
net.MLP.prototype.optimize_the_parameters = function( the_gradients_for_weights={}, 
                                                      the_gradients_for_bias={} 
){
    let modelContext = this; //The model context

    //For each layer
    modelContext.getLayers().forEach(function( current_layer, 
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
                            .subtract( modelContext.learning_rate * calculated_gradients_values_for_weight );

            });

            let calculated_gradients_values_for_bias = the_gradients_for_bias[`layer${ layer_index }`][ `unit${ unit_index }` ];

            //Update bias
            current_unit.subtractBias( modelContext.learning_rate * calculated_gradients_values_for_bias );

        });

    });
}