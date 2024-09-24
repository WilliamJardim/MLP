/**
* A utility used for accumulate the LOSSES in a hidden layer
* This accumulation is made using a sum 
*/
net.LOSS_Accumulator = function(){
    let context = {};
    context._acumulated = 0;

    /** 
    * Compute the derivative using the chain rule AND sum in the __acumulated 
    * 
    * @param {Number} eloh_param  - The parameters what connect the unit
    * @param {Number} LOSS        - The unit LOSS
    */
    context.accumulate = function( eloh_param=Number(), 
                                   LOSS=Number() 
    ){
        let thisDerivative  = eloh_param * LOSS;
        context._acumulated = context._acumulated + thisDerivative;
    }

    /**
    * Get the accumulated value
    * @returns {Number} - the accumulated value
    */
    context.getAccumulatedValue = function(){
        return context._acumulated;
    }

    return context;
}