validations = {};

//A maior quantidade de features no dataset
validations.getMostFeaturesLength = function( datasetCheck ){
    let lengths = [];
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        lengths.push( datasetCheck[lc][0].length );    
    }

    let maxLen = lengths[0];
    for( let lc = 0 ; lc < maxLen.length ; lc++ )
    {
        if( lengths[lc] > maxLen ){
            maxLen = lengths[lc];
        }   
    }

    return maxLen;
}

validations.throwErrorIfSomeSampleAreIncorrectArrayLength = function(datasetCheck){
    let maxLength = validations.getMostFeaturesLength( datasetCheck );
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        if( datasetCheck[lc][0].length != maxLength ){
            throw Error(`Some sample in dataset are different size of the others!. The dataset samples must have same sizes!`);
        }
    }
}

validations.throwErrorIfSomeSampleAreDiffentLengthOfInputsThatTheInputLayer = function( inputs_amount, datasetCheck ){
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        if( datasetCheck[lc][0].length != inputs_amount ){
            throw Error(`Some sample in dataset are different size of the inputs in the input layer!`);
        }
    }
}