//A Layer
net.Layer = function( layer_config={}, afterCreateCallback=()=>{} ){
    let context = {};

    context.objectName            = 'Layer';
    context.layer_config          = layer_config;
    context.number_of_units       = layer_config.units;
    context.amount_of_estimatives = context.number_of_units; //the same as context.number_of_units
    context.number_of_inputs      = layer_config.inputs;
    context.activation_function   = layer_config.activation;
    context.layer_type            = layer_config.type;
    context.afterCreateCallback   = afterCreateCallback;

    //External vincules
    context.model_parameters      = layer_config.model_parameters;

    context.vincules              = {}; //Store all vincules
    
    //The units objects that will be created below
    context.units        = [];

    /**
    * Get the layer father(the model)
    */
    context.getFather = function(){
        return context._father;
    }

    /**
    * Get the index of this layer 
    */
    context.getLayerIndex = function(){
        return context._internal_index;
    }

    /**
    * Add a unit to context.units 
    * 
    * @param {net.Layer} new_unit_object
    */
    context.addUnit = function( new_unit_object ){

        if( typeof new_unit_object == 'object' && 
            new_unit_object instanceof Object &&
            new_unit_object.objectName == 'Unit'
        ){
            context.units = [...context.units, new_unit_object];

        }else{
            throw Error('The "new_unit_object" is not a object of type Unit.');
        }

    }

    /**
    * Get the own context
    * @returns {Object} - the layer it self
    */
    context.getSelfContext = function(){
        return context;
    }

    /**
    * Get the own context
    * @returns {Object} - the layer it self
    */
    context.atSelf = context.getSelfContext;

    /**
    * Vinculate a prop to this layer
    * @param {String}   newAttributeName
    * @param {any}      valueOfThisAttribute
    * @returns {Object} - Layer It self
    */
    context.vinculate = function(newAttributeName, valueOfThisAttribute){
        context[ newAttributeName ] = valueOfThisAttribute;
        context.vincules[ newAttributeName ] = context[ newAttributeName ];
        return context;
    }

    /**
    * Get the layer number(in model) 
    */
    context.getIndex = function(){
        return context['_internal_index'];
    }

    /**
    * Return the next layer( That is, it returns the layer that comes after the current layer )
    * @returns {net.Layer}
    */
    context.getNextLayer = function( callback_in_layer=null ){
        let next_layer_index = context.getIndex() + 1;

        let layer_it_context = context.getFather()
                                      .getLayer( next_layer_index ) || null;

        if( callback_in_layer )
        {
            callback_in_layer.bind(layer_it_context)( layer_it_context, next_layer_index );
        }
        
        return layer_it_context;
    }

    /**
    * Return the previous layer( That is, it returns the layer that comes before the current layer )
    * @returns {net.Layer}
    */
    context.getPreviousLayer = function(){
        let previous_layer_index = context.getIndex() - 1;

        return context.getFather()
                      .getLayer( previous_layer_index ) || null;
    }

    /**
    * Check if this layer is of type 
    */
    context.is = function( layerType ){
        return context.layer_type == layerType ? true : false;
    }

    /**
    * Check if this layer NOT IS of type 
    */
    context.notIs = function( layerType ){
        return !context.is( layerType );
    }

    /**
    * A getter for context.units
    */
    context.getUnits = function(){
        return context.units;
    }

    /**
    * Get a unit in this layer
    * @param {Number} unit_index 
    * @returns {Object}
    */
    context.getUnit = function( unit_index ){
        return context.units[ unit_index ];
    }

    /**
    * Set the inputs of this layer, to be used in context.get_Estimative_of_Units
    */
    context.setInputs = function( LAYER_INPUTS=[] ){
        let this_layer_ref        = context,
            this_layer_index      = this_layer_ref.getIndex(),
            father_ref            = this_layer_ref.getFather(),
            inputs_of_each_layer  = father_ref.readProp('inputs_of_each_layer');  

        if( !(LAYER_INPUTS instanceof Array) ){
            throw Error(`LAYER_INPUTS need be a Array!`);
        }

        /**
        * Assign the LAYER_INPUTS in the prop "inputs_of_each_layer"
        */
        inputs_of_each_layer[ `layer${ this_layer_index }` ] = [... LAYER_INPUTS.copyWithin()];

        /*
        * Create a direct reference to make easy access this via layer
        */
        this_layer_ref[ 'LAYER_INPUTS' ] = inputs_of_each_layer[ `layer${ this_layer_index }` ];
    }

    /**
    * Get the inputs of this layer, to be used in context.get_Estimative_of_Units
    */
    context.getInputs = function(){
        return context['LAYER_INPUTS'];
    }

    /** 
    * Get the estimated value of each unit in the current layer 
    */
    context.get_Estimative_of_Units = function(){

        /**
        * Get this layer inputs( that was linked to this object )
        * Remembering that the inputs of this layer are the estimated values of the previous layer
        */
        let LAYER_INPUTS = context.getInputs();

        /**
        * For each unit in this layer <layer_index>, get the UNIT ESTIMATED VALUE and store inside the unit
        */
        let units_estimatives = [];

        /**
        * Compute the estimated value of each unit in this layer 
        */
        context.getUnits().forEach(function( current_unit ){

            let unit_estimated_data  = current_unit.estimateValue( LAYER_INPUTS );
            let act_potential        = unit_estimated_data.unit_potential;
            let unit_estimative      = unit_estimated_data.activation_function_result;

            units_estimatives.push( unit_estimative );
        });

        return units_estimatives;
    }

    //Run the callback
    context.afterCreateCallback.bind(context)( context );

    //Initialize the layer
    let i = 0;
    while( i < context.number_of_units )
    {
        /**
        * Add a unit to the context.units, using a Unit callback
        */
        new net.Unit({
           number_of_inputs     : context.number_of_inputs,
           activation_function  : context.activation_function

        }, ( unitItSelf )=>{

                const layer_context = context.getSelfContext();
                const unit_context  = unitItSelf.getSelfContext();

                /**
                * Do important vincules 
                */
                unit_context.atSelf()
                            .vinculate( '_unitIndex',      i);

                unit_context.atSelf()
                            .vinculate( '_layerRef' ,      context );

                unit_context.atSelf()
                            .vinculate('model_parameters', context.model_parameters);

                /**
                * Generate parameters for this unit
                */
                unit_context.atSelf()
                            .generate_random_parameters( context.number_of_inputs );

                /* Add the unit in the layer */
                layer_context.atSelf()
                             .addUnit( unit_context );

        });

        //Create next unit in next iteration
        i++;

    }

    /**
    * Return the layer ready
    */
    return context;
}