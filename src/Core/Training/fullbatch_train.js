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
    
    let currentEpoch_fullback = 0;
    let last_total_loss_fullbatch = 0;
    let loss_history_fullbatch    = [];

    //While the current epoch number is less the number_of_epochs
    while( currentEpoch_fullback <= number_of_epochs )
    {
        let total_loss_fullbatch = 0;

        //Accumulate the batch and update the parameters
        context.accumulate_batch_gradients( train_samples );
        
        total_loss_fullbatch += context.compute_train_cost( train_samples );

        last_total_loss_fullbatch = total_loss_fullbatch;
        loss_history_fullbatch.push(total_loss_fullbatch);

        if( String( Number(currentEpoch_fullback / 100) ).indexOf('.') != -1 ){
            console.log(`LOSS: ${last_total_loss_fullbatch}, epoch ${currentEpoch_fullback}`)
        }

        //Goto next epoch
        currentEpoch_fullback++;
    }

    return {
        last_total_loss: last_total_loss_fullbatch,
        loss_history: loss_history_fullbatch
    };
}