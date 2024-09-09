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
            throw Error(`Some sample in dataset have different size of the others!. The dataset samples must have same sizes!`);
        }
    }
}

validations.throwErrorIfSomeSampleAreDiffentLengthOfInputsThatTheInputLayer = function( inputs_amount, datasetCheck ){
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        if( datasetCheck[lc][0].length != inputs_amount ){
            throw Error(`Some sample in dataset have different size of the inputs in the input layer!`);
        }
    }
}

validations.throwErrorIfSomeSampleAreStringsOrCharacters = function(datasetCheck){
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        if( datasetCheck[lc][0].some( function( value ){ return typeof value != 'number' } ) ||
            datasetCheck[lc][1].some( function( value ){ return typeof value != 'number' } )
        ){
            throw Error(`Some sample in dataset have strings values or objects instead numbers!`);
        }
    }
}

validations.throwErrorIfSomeSampleHaveObjectsArraysInsteadValues = function(datasetCheck){
    for( let lc = 0 ; lc < datasetCheck.length ; lc++ )
    {
        if( (typeof datasetCheck[lc][0] == 'object' && !(datasetCheck[lc][0] instanceof Array) ) || (typeof datasetCheck[lc][1] == 'object' && !(datasetCheck[lc][1] instanceof Array)) ){
            throw Error(`Some sample have Objects or Arrays. The sample values cannot be Objects neither Arrays!`);
        }

        if( datasetCheck[lc][0].some( function( value ){ return (value instanceof Array) || typeof value == 'object'} ) ||
            datasetCheck[lc][1].some( function( value ){ return (value instanceof Array) || typeof value == 'object' } )
        ){
            throw Error(`Some sample have Objects or Arrays. The sample values cannot be Objects neither Arrays!`);
        }
    }
}
