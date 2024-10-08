/**
* SGD/Online training
* Update the weights after each individual example
* 
* @param {Array} train_samples 
* @param {Array} number_of_epochs 
* @returns {Object}
*/
net.MLP.prototype.online_train = function( train_samples, 
                                           number_of_epochs
){
    let modelContext = this; //The model context

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
            let sample_desired_value    = sample_data[1]; //SAMPLE DESIRED VALUES

            //Validations before apply the backpropagation
            if( !(sample_features instanceof Array) ){
                throw Error(`The variable sample_features=[${sample_features}] must be a Array!`);
            }

            if( !(sample_desired_value instanceof Array) ){
                throw Error(`The variable sample_desired_value=[${sample_desired_value}] is not a Array!`);
            }

            //If the number of items in the sample_desired_value Array is different from the number of units in the final layer
            if( sample_desired_value.length != modelContext.layers[ modelContext.layers.length-1 ].units.length ){
                throw Error(`The sample_desired_value=[${sample_desired_value}] has ${sample_desired_value.length} elements, But must be ${modelContext.layers[ modelContext.layers.length-1 ].units.length}(the number of units in final layer)`);
            }

            //Do backpropagation and Gradient Descent
            let calculated_gradients_data = modelContext.backpropagate_sample({
                sample_inputs  : sample_features, 
                desiredValuess : sample_desired_value, 

                /**
                * Callback called before the "backpropagate_sample" execution
                */
                beforeThis({}){
                    //Nothing
                },
                
                /**
                * Callback called after the "backpropagate_sample" execution
                */
                afterThis({ modelContext,
                            gradients_per_layer,
                            gradients_of_each_unit_weights_per_layer,
                            gradients_of_each_unit_bias_per_layer
                }){
                    let gradients_for_weights  = gradients_of_each_unit_weights_per_layer;
                    let gradients_for_bias     = gradients_of_each_unit_bias_per_layer;

                    /**
                    * Applies the Gradient Descent algorithm to update the parameters
                    */
                    modelContext.optimize_the_parameters( gradients_for_weights, gradients_for_bias );
                }
            });

        });

        total_loss += modelContext.compute_train_cost( train_samples );

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