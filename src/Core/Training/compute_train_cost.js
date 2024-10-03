/**
* Compute de COST
* 
* @param {Array} train_samples - The training samples
* @returns {Number} - The cost
*/
net.MLP.prototype.compute_train_cost = function( train_samples ){

    let modelContext = this; //The model context

    //Validations
    //If is a array or a instance of net.data.Dataset, is acceptable and compative type
    if( !(train_samples instanceof Array || train_samples.internalType == 'dataset') ){
        throw Error(`The train_samples=[${train_samples}] need be a Array instance!`);
    }

    if( train_samples.length == 0 ){
        throw Error(`The train_samples is a empty Array!`);
    }

    let cost = 0;
    
    train_samples.forEach(function( sample_data ){  

        let sample_features        = sample_data[0]; //SAMPLE FEATURES
        let sample_desired_value   = sample_data[1]; //SAMPLE DESIRED VALUES
        
        let estimatedValues        = modelContext.estimate_values(sample_features)
                                                 .getEstimatedValues();

        for( let S = 0 ; S < estimatedValues.length ; S++ )
        {
            cost = cost + ( sample_desired_value[ S ] - estimatedValues[ S ] ) ** 2;
        }

    });

    return cost;
}