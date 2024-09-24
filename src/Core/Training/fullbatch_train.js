/**
* Full Batch training
* Update the weights just one time per epoch. 
* 
* @param {Array} train_samples 
* @param {Array} number_of_epochs 
* @returns {Object}
*/
net.MLP.prototype.fullbatch_train = function( train_samples, 
                                              number_of_epochs 
){
    let context = this; //The model context
    
    let currentEpoch    = 0;
    let last_total_loss = 0;
    let loss_history    = [];

    //While the current epoch number is less the number_of_epochs
    while( currentEpoch < number_of_epochs )
    {
        let total_loss = 0;

        //Accumulate the batch and update the parameters
        context.accumulate_batch_gradients( train_samples );

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