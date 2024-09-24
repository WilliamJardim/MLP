/**
* Mini Batch training
* Update the weights after every batch. 
* 
* @param {Array} train_samples 
* @param {Array} number_of_epochs 
* @returns {Object}
*/
net.MLP.prototype.minibatch_train = function( train_samples, 
                                              number_of_epochs, 
                                              samples_per_batch=2 
){
    let context = this;//The model context

    let currentEpoch    = 0;
    let last_total_loss = 0;
    let loss_history    = [];

    //Split into N mini batches
    let sub_divisions = [];
    let current_set   = [];

    train_samples.forEach(function( sample_obj, 
                                    sample_index
    ){

        if( current_set.length < samples_per_batch )
        {
            current_set.push( sample_obj );

        //If the current set reach the samples_per_batch limit
        }else{
            sub_divisions.push( [... current_set.copyWithin()] );
            current_set = [];
        }

    });

    //While the current epoch number is less the number_of_epochs
    while( currentEpoch < number_of_epochs )
    {
        let total_loss = 0;

        //For each division
        sub_divisions.forEach(function( actual_train_set ){

            /**
            * Accumulate the gradients of each weight of each unit of each layer
            * Then, update the weights and bias in the final of the batch 
            */
            context.train_fullbatch( actual_train_set );

        });

        total_loss += context.compute_train_cost( train_samples );

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