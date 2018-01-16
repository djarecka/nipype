import pdb


def ordering(el, i, current_sign=None):
    global output_mapper
    if type(el) is tuple:
        iterate_list(el, ".")
    elif type(el) is list:
        iterate_list(el, "*")
    elif type(el) is str:
        output_mapper.append(el)
    else:
        raise Exception("WRONG input")
    
    if i > 0:
        output_mapper.append(current_sign)


def iterate_list(element,  sign):
    for i, el in enumerate(element):
        ordering(el, i, current_sign=sign)


def mapper2rpn(mapper):
    global output_mapper
    output_mapper = []
    
    ordering(mapper, i=0)
    
    #print("mapper", output_mapper)
    return output_mapper


def mapping_axis(state_inputs, mapper_rpn):
    axis_for_input = {}
    stack = []
    current_axis = None
    current_shape = None
    
    for el in mapper_rpn:
        if el == ".":
            right = stack.pop()
            left = stack.pop()
            if left == "OUT":
                if state_inputs[right].shape == current_shape: #todo:should we allow for one-element array? 
                    axis_for_input[right] = current_axis
                else:
                    raise Exception("arrays for scalar operations should have the same size")

            elif right == "OUT":
                if state_inputs[left].shape == current_shape:
                    axis_for_input[left] = current_axis
                else:
                    raise Exception("arrays for scalar operations should have the same size")

            else:
                #pdb.set_trace()
                if state_inputs[right].shape == state_inputs[left].shape:
                    current_axis = list(range(state_inputs[right].ndim))
                    current_shape = state_inputs[left].shape
                    axis_for_input[left] = current_axis
                    axis_for_input[right] = current_axis
                else:
                    raise Exception("arrays for scalar operations should have the same size")
                
            stack.append("OUT")

        elif el == "*":
            right = stack.pop()
            left = stack.pop()
            if left == "OUT":
                axis_for_input[right] = [i + 1 + current_axis[-1] for i 
                                              in range(state_inputs[right].ndim)]
                current_axis = current_axis + axis_for_input[right]
                current_shape = tuple([i for i in current_shape + state_inputs[right].shape])
            elif right == "OUT":
                for key in axis_for_input:
                    #pdb.set_trace()
                    axis_for_input[key] = [i + state_inputs[left].ndim for i 
                                           in axis_for_input[key]]

                axis_for_input[left] = [i-len(current_shape) + current_axis[-1] +1 for i 
                                        in range(state_inputs[left].ndim)]
                current_axis = current_axis + [i + 1 + current_axis[-1] for i 
                                               in range(state_inputs[left].ndim)]
                current_shape = tuple([i for i in state_inputs[left].shape + current_shape])
            else:
                axis_for_input[left] = list(range(state_inputs[left].ndim))
                axis_for_input[right] = [i+state_inputs[left].ndim for i 
                                              in range(state_inputs[right].ndim)]
                current_axis = axis_for_input[left] + axis_for_input[right]
                current_shape = tuple([i for i in 
                                       state_inputs[left].shape+state_inputs[right].shape])
            stack.append("OUT")

        else:
            stack.append(el)

    if len(stack) > 1:
        raise Exception("exception from mapping_axis")
    elif stack[0] != "OUT":
        #pdb.set_trace()
        current_axis = [i for i in range(state_inputs[stack[0]].ndim)]
        axis_for_input[stack[0]] = current_axis

    #print("axis", axis_for_input)
    return axis_for_input, max(current_axis) + 1



def converting_axis2input(state_inputs, axis_for_input, ndim):
    input_for_axis = []
    shape = []
    for i in range(ndim):
        input_for_axis.append([])
        shape.append(0)
        
    for inp, axis in axis_for_input.items():
        for (i, ax) in enumerate(axis):
            input_for_axis[ax].append(inp)
            shape[ax] = state_inputs[inp].shape[i]
            
    return input_for_axis, shape
