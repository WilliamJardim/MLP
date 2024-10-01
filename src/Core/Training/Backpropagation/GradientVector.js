/**
* Compute the Partial Derivative of each weight of a unit
* @param {Number} loss_wrt_unit_estimation - The LOSS with respect to unit estimation-function
* @param {unit_weights} unit_weights       - The unit weights
* @returns {Object}
*/
net.GradientVector = function( {loss_wrt_unit_estimation, unit_inputs} ){
    let context = {};
    context.loss_wrt_unit_estimation  = loss_wrt_unit_estimation;
    context._inputs_of_unit           = unit_inputs; 

    context.loss_wrt_unit_weights = [];

    /**
    * Compute LOSS with respect to each weight
    */
    for( let i = 0 ; i < context._inputs_of_unit.length ; i++ )
    {
        let current_weight_input  = context._inputs_of_unit[ i ];

        /** 
        * Apply the chain rule
        */
        let external_derivative  = context.loss_wrt_unit_estimation;

        /**
        * The partial derivative with respect to the a weight is their input, because, the partial derivative of a parameter with respect to the parameter itself is one.
        */
        let internal_derivative  = external_derivative * current_weight_input; 

        /** 
        * Store this derivative value
        */
        context.loss_wrt_unit_weights[ i ] = internal_derivative; 
    }

    context.get_wrt_unit_estimation = function(){
        return context.loss_wrt_unit_estimation;
    }

    return context;
}