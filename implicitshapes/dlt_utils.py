import logging
import yaml
import os
import shutil
import math
from enum import Enum
from tqdm import tqdm
from time import gmtime, strftime

from collections import OrderedDict
from easydict import EasyDict as edict

import torch
import numpy as np
from flatdict import FlatDict

import nibabel as nib

CALLBACK_FN_NAMES = [
    'callback_pre_train_epoch',
    'callback_post_train_epoch',
    'callback_pre_val_epoch',
    'callback_post_val_epoch',
    'callback_pre_train_iter',
    'callback_post_train_iter',
    'callback_pre_val_iter',
    'callback_post_val_iter',
]

# adapted from
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py
# should be able to load a state dict that was wrapped with DataParallel. Should also be
# able to load into a wrapped DataParallel model
def load_model_from_ckpt(model, ckpt_path, partial=False, partial_ckpt=False):
    """
    When partial = True, the specified model is allowed to have weights that are not
    present in the saved ckpt. Thus it will only load the subset of weights within the
    ckpt.
    When partial_ckpt = True, this allows the reverse, i.e., the ckpt to have weights not
    present in the specified model
    """

    loaded = torch.load(ckpt_path)
    model_state_dict = loaded['component.model']

    new_state_dict = OrderedDict()
    # strip 'module.' from every key if applicable
    for k,v in model_state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module prefix
        new_state_dict[k] = v


    if partial_ckpt:
        new_state_dict = load_partial_ckpt(model, new_state_dict)

    if not partial:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
    # if we are using partial loading
    else:
        if isinstance(model, torch.nn.DataParallel):
            # get the specified model state dict
            cur_model_dict = model.module.state_dict()
            # update it with the loaded weights
            cur_model_dict.update(new_state_dict)
            # load it into the specified model
            model.module.load_state_dict(cur_model_dict)
        else:
            cur_model_dict = model.state_dict()
            cur_model_dict.update(new_state_dict)
            model.load_state_dict(cur_model_dict)

    return model


def read_yaml(path):
    try:
        with open(path, "r") as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        return None

def dlt_print(msg, level=logging.INFO):
    """
    Simple util function to both print to console and log
    """
    print(msg)
    logging.log(level, msg)


def init_workspace(ws_path):
    if os.path.exists(ws_path):
        shutil.rmtree(ws_path)
    os.makedirs(ws_path)

"""
Base transform class
"""


class BaseTransform(object):
    def __init__(self, fields):
        assert (isinstance(fields, (str, list))), "Fields must be a string or a list of strings"

        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields

    def __call__(self, sample):
        assert (isinstance(sample, dict) or isinstance(sample, Context)), "Each sample must be a dict or Context"


class Callable(BaseTransform):
    """
    Apply a user-defined callable object (e.g., function or torchvision transform) as transform.
    kw_args allows you to specify arguments, e.g., for np.stack
    Callable(fields='image', callable_object=np.stack, axis=1)
    """

    def __init__(self, fields, callable_object, **kw_args):
        """
        Args:
            callable_object (callable): Lambda/function to be used for transform.
            kw_args: additional named arguments to send to callable
        """
        super().__init__(fields)
        assert callable(callable_object), repr(type(callable_object).__name__) + " object is not callable"
        self._callable_object = callable_object
        self.perm_args = kw_args

    def __call__(self, data_dict):
        for field in self.fields:
            data_dict[field] = self._callable_object(data_dict[field], **self.perm_args)

        return data_dict

class CenterIntensities(BaseTransform):
    """
    Transform that subtracts by a subtrahend and divides by a divisor, most
    often done to whiten data by subtracting the mean and dividing by the std
    deviation.

    Note, this class assumes the pytorch shape conventions:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    which consider the first dimension (index 0) to be the channel dimension.
    """

    def __init__(self, fields, subtrahend, divisor=1.0):
        """
        Args:
            fields: fields to apply centering
            subtrahend: the subtrahend used to subtract, if a list then subtraction
                is performed per channel
            divisor: sames as subtrahend, but specifies the divisor.
        """
        super().__init__(fields)

        # convert any lists to np.arrays, with an extra singleton dimension
        # to allow broadcasting
        if isinstance(divisor, list):
            divisor = np.array(divisor)
            divisor = np.expand_dims(divisor, 1)
        if isinstance(subtrahend, list):
            subtrahend = np.array(subtrahend)
            subtrahend = np.expand_dims(subtrahend, 1)
        self.subtrahend = subtrahend
        self.divisor = divisor

    def __call__(self, data_dict):

        for field in self.fields:
            old_shape = data_dict[field].shape

            # reshape val, to allow broadcasting over 2D, 3D, or nd data
            val = data_dict[field].reshape((data_dict[field].shape[0], -1))

            # perform centering
            val -= self.subtrahend
            val /= self.divisor
            data_dict[field] = val.reshape(old_shape)

        return data_dict

class Clip(BaseTransform):
    """
    Will clip numpy arrays and pytorch tensors
    """
    def __init__(self, fields, new_min=0.0, new_max=1.0):
        """
        new_min: min value to clip to
        new_max: max value to clip to
        """
        super().__init__(fields)

        self._new_min = new_min
        self._new_max = new_max

    def __call__(self, data_dict):

        for field in self.fields:
            val = data_dict[field]
            # check if numpy or torch.Tensor, and call appropriate method
            if isinstance(val, torch.Tensor):
                data_dict[field] = torch.clamp(val, self._new_min, self._new_max)
            else:
                data_dict[field] = np.clip(val, self._new_min, self._new_max)

        return data_dict

class ExpandDims(BaseTransform):
    """
    Adds a dimension to specified axis, works for both numpy arrays and pytorch
    tensors
    """

    def __init__(self, fields, axis=0):
        super().__init__(fields)

        self._axis = axis

    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:
            val = data_dict[field]
            if isinstance(val, torch.Tensor):
                data_dict[field] = torch.unsqueeze(val, self._axis)
            else:
                data_dict[field] = np.expand_dims(val, self._axis)

        return data_dict

class NiBabelLoader(BaseTransform):
    """
    Loads an nibabel volume, given a set of fields given as paths
    For each field, will also add a new field appended with '_meta' that
    contains affine transform and nibabel header information
    """


    def __init__(self, fields, root_dir=None, dtype=np.float32):

        if root_dir is not None:
            assert(type(root_dir)==str), "Root dir must be string"

        super().__init__(fields)

        self._root_dir = root_dir
        self._dtype = dtype

    def __call__(self, sample):

        super().__call__(sample)

        for field in self.fields:
            path = sample[field]
            if self._root_dir:
                path = os.path.join(self._root_dir, path)
            img = nib.load(path)
            # overwrite the path with the numpy array
            img_data = img.get_fdata().astype(self._dtype)
            sample[field] = img_data
            # create a dict for meta info, with fields for affine and one for header
            meta = {}
            meta['affine'] = img.affine
            # copy header information in a native dict, that is compatible with pytorch batching
            meta['header'] = {}
            meta['bitpix'] = img.header['bitpix']
            meta['dim'] = img.header['dim']
            meta['pixdim'] = img.header['pixdim']
            meta['scl_slope'] = img.header['scl_slope']
            meta['scl_inter'] = img.header['scl_inter']
            sample[field+'_meta'] = meta

        return sample

class Component(object):
    """Base class for all components that uses the callback mechanism
    """

    def __init__(self):
        """Constructor of Component object

        Args:
            dependency: dependency for all callbacks of this component
        """
        self.context_key = None
        self._dependencies = {}

    def register(self, context, key):
        self.context_key = key
        context[key] = self

    def get_callback(self, fn_name):
        """Get callback function with the given name and its dependency
        """
        if hasattr(self, fn_name):
            fn = getattr(self, fn_name)
            dependency = self._dependencies.get(fn_name, [])
        else:
            fn = None
            dependency = []

        return fn, dependency

    def add_dependency(self, fn_name, component_key):
        """Add callback dependency
        Args:
            fn_name: the callback function to be processed
            component_key: the context key of the component or the component instance
                to be added to dependency
        """

        if isinstance(component_key, Component):
            component_key = component_key.context_key

        if not isinstance(component_key, str):
            raise ValueError(
                'The 2nd argument of add_dependency should be component key (string) in context. Got type {} instead!'.format(
                    type(component_key)))

        # If fn_name is `all`, add dependency for all callback
        if fn_name == 'all':
            for key in CALLBACK_FN_NAMES:
                self.add_dependency(key, component_key)
            return

        if fn_name not in CALLBACK_FN_NAMES:
            raise ValueError('Unrecognized callback function name: {}'.format(fn_name))

        # Create list if not exist
        if fn_name not in self._dependencies:
            self._dependencies[fn_name] = []

        # Add dependency only if it does not exist
        if component_key not in self._dependencies[fn_name]:
            self._dependencies[fn_name].append(component_key)

    def remove_dependency(self, fn_name, component_key):
        """Remove callback dependency
        Args:
            fn_name: the callback function to be processed
            component_key: the context key of the component to be removed from the dependency
        """

        if isinstance(component_key, Component):
            component_key = component_key.context_key

        if not isinstance(component_key, str):
            raise ValueError(
                'The 2nd argument of remove_dependency should be component key (string) in context. Got type {} instead!'.format(
                    type(component_key)))

        # If fn_name is `all`, remove dependency for all callback
        if fn_name == 'all':
            for key in CALLBACK_FN_NAMES:
                self.remove_dependency(key, component_key)
            return

        if fn_name not in CALLBACK_FN_NAMES:
            raise ValueError('Unrecognized callback function name: {}'.format(fn_name))

        if fn_name not in self._dependencies:
            return

        if component_key not in self._dependencies[fn_name]:
            return

        self._dependencies[fn_name].remove(component_key)

    def clear_dependency(self, fn_name):
        """Clear all callback dependency
        Args:
            fn_name: the callback function to be processed
        """
        # If fn_name is `all`, clear dependency for all callback
        if fn_name == 'all':
            for key in CALLBACK_FN_NAMES:
                self.clear_dependency(key)
            return

        if fn_name not in CALLBACK_FN_NAMES:
            raise ValueError('Unrecognized callback function name: {}'.format(fn_name))

        if fn_name not in self._dependencies:
            return

        self._dependencies.pop(fn_name)


class MoveToCuda(BaseTransform):
    """
    Moves specified tensors to cuda
    """

    def __init__(self, fields, non_blocking=False):
        super().__init__(fields)
        self.non_blocking = non_blocking

    def __call__(self, data_dict):
        for field in self.fields:
            data_dict[field] = data_dict[field].cuda(non_blocking=self.non_blocking)

        return data_dict


class PreIterTransform(Component):
    def __init__(self, transform, do_train=True, do_val=True):
        super().__init__()

        self._transform = transform
        self.do_val = do_val
        self.do_train = do_train

    def callback_pre_train_iter(self, context):
        if self.do_train:
            self._transform(context)

    def callback_pre_val_iter(self, context):
        if self.do_val:
            self._transform(context)


class Context(object):
    """
    Class for holding concepts.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._data = FlatDict(delimiter=".")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, value=None):
        return self._data.get(key, value)

    def __setitem__(self, key, val):
        self._data[key] = val
        if isinstance(val, Component):
            val.context_key = key

    def __iter__(self):
        return iter(self._data)

    def increment(self, key):
        self._data[key] += 1

    def __repr__(self):
        ret_str = "dlt.common.core.Context\nComponents:\n"

        for k, v in self._data.items():
            if "component" in k:
                ret_str += repr(v) + "\n"

        return ret_str

    def pop(self, key, *args):
        """
        Follows dict.pop convetion: if second parameter is specifed, then returns val
        if key is not in the dict, avoiding error if it's not there
        """
        return self._data.pop(key, *args)

    def state_dict(self):
        """
        Generate a state dictonary of the current context.
        The state dictionary is a dictionary of {item_key: item_state}
        For component items, their state_dict() is called to generate their item_state, if
        they have one. Otherwise the item_state is None.
        For variable items, the item_state is themselves (we assume it can be pickled for saving).
        """
        state_dict = {}
        for key in self._data:
            if key.startswith("component"):
                state_dict_fn = getattr(self._data[key], "state_dict", None)
                if callable(state_dict_fn):
                    state_dict[key] = state_dict_fn()
                else:
                    state_dict[key] = None
            else:
                state_dict[key] = self._data[key]

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load a state dictonary into context.
        The state dictionary is a dictionary of {item_key: item_state}
        For component items: their load_state_dict() is called to load the the item_state back.
        For variable items: the item_state is assigned to them.
        """
        for key in state_dict:
            if key.startswith("component"):
                if key not in self._data:
                    logging.info(
                        "'{}' not found in context, skip loading state_dict".format(key)
                    )
                    continue

                if state_dict[key] is None:
                    logging.info(
                        "state_dict for '{}' is empty, skip loading.".format(key)
                    )
                    continue

                load_state_dict_fn = getattr(self._data[key], "load_state_dict", None)
                if callable(load_state_dict_fn):
                    logging.info("Loading state_dict for '{}'".format(key))
                    load_state_dict_fn(state_dict[key])
                else:
                    logging.warning(
                        "'{}' does not have load_state_dict method, skip loading.".format(
                            key
                        )
                    )
            else:
                logging.info("Loading state_dict for '{}'".format(key))
                self._data[key] = state_dict[key]

        logging.info("Context state_dict is loaded into context.")

    def save(self, f):
        try:
            torch.save(self.state_dict(), f)
            logging.info("Context state_dict is saved to {}.".format(f))
        except:
            error_msg = "At least one item in the state_dict cannot be pickled"
            raise ValueError(error_msg)

    def load(self, f):
        logging.info("Context state_dict is loaded from {}.".format(f))
        self.load_state_dict(torch.load(f))


def _process_component(context, key, fn_name, fn_list, processing_stack):
    """A recursive function to add callback into list following dependencies"""
    fn, dependency = context[key].get_callback(fn_name)

    # No need to process if fn does not exist or already processed
    if fn is None or fn in fn_list:
        return fn_list, processing_stack

    # if component is already in the stack (meaining it's wainting for its dependencies to
    # be processed), then there is a dependency loop. Throw exception.
    if key in processing_stack:
        processing_stack.append(key)
        error_msg = "Callback dependency loop detected! Stack: {}".format(
            processing_stack
        )
        raise ValueError(error_msg)

    # Add the component to the stack under processing
    processing_stack.append(key)

    for item in dependency:
        if item not in context:
            error_msg = "'{}' depends on '{}', which does not exist in context!".format(
                key, item
            )
            raise ValueError(error_msg)

        fn_list, processing_stack = _process_component(
            context, item, fn_name, fn_list, processing_stack
        )

    fn_list.append(fn)
    logging.info(
        "Collected {} function #{} from '{}': {} (dependency: {})".format(
            fn_name, len(fn_list), key, type(context[key]), dependency
        )
    )

    # Remove the component from the stack under process
    processing_stack.remove(key)

    return fn_list, processing_stack


def collect_callback_fn(context, fn_name):
    """
    Collect a list of callback functions respecting the dependencies
    Args:
        context: context object
        fn_name: name of callback function to be collected

    Return:
        master_callback: a master callback function that calls all collected callbacks
    """
    logging.info("START COLLECTING {} CALLBACK FUNCTIONS".format(fn_name))
    if fn_name not in CALLBACK_FN_NAMES:
        error_msg = "Unrecognized callback function name: {}".format(fn_name)
        raise ValueError(error_msg)

    fn_list = []
    processing_stack = []

    # Go over objects in the context, and add the callback functions they have to a list
    for key in context:

        # Only search callbacks in components
        if not key.startswith("component."):
            continue

        # Skip if the object is not derived from Component class
        if not isinstance(context[key], Component):
            continue

        fn_list, processing_stack = _process_component(
            context, key, fn_name, fn_list, processing_stack
        )

    def master_callback(ct):
        """A master callback functions that calls all callbacks"""
        for cb_fn in fn_list:
            cb_fn(ct)

    return master_callback




class MonitorComponent(Component):
    """
    Monitor component has low default priority because nothing should depend on it
    """

    def __init__(self):
        super().__init__()


class ConsoleLogger(MonitorComponent):
    """
    Will log the epoch, iter, and losses automatically.
    You can optionally provide additional print outs in the form of a list of
    tuples, (var_name, string_format).
    E.g., if you are computing average loss you can provide a tuple of
    ('var.avg_train_loss', 'Avg Train Loss: %.3f'), which will add to the
    log

    The format for the per-iter logging is 'Epoch: %d, Iter: %d, Train Loss: %.3f',
    with any extra (variable,format) tuples append to the default line


    """

    def __init__(self, extra_train_logs=None, extra_val_logs=None):
        """
        All args should be a list (or singleton) of tuples, (var_tag, format_string),
        with var_tag specifying the variable tag in the context and format string
        specifying how you want it formattted, e.g., "Awesome Variable %.3f" for a float.
        Only put in one string format argument (%) per var_tag
        """
        super().__init__()
        # turn everything into lists if they aren't already
        if not extra_train_logs:
            extra_train_logs = []
        if not isinstance(extra_train_logs, list):
            extra_train_logs = [extra_train_logs]
        if not extra_val_logs:
            extra_val_logs = []
        if not isinstance(extra_val_logs, list):
            extra_val_logs = [extra_val_logs]

        self._extra_train_logs = extra_train_logs
        self._extra_val_logs = extra_val_logs

        # now set up the string with the format arguments
        # for per-iter logging, we do one giant line, startign with epoch, iter
        # and loss information
        self._train_format = '[%s] Epoch: %d, Iter: %d, Train Loss: %.3f'
        # then we add any extra variable string logs, if specified
        for extra_train_log in self._extra_train_logs:
            self._train_format += ", " + extra_train_log[1]
        # do the same for validation
        self._val_format = '[%s] Epoch: %d, Iter: %d, Val Loss: %.3f'
        for extra_val_log in self._extra_val_logs:
            self._val_format += ", " + extra_val_log[1]

    def _post_train_iter_string(self, context):
        # grab the mandatory variables from the context
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        current_epoch = context['var.current_epoch']
        current_iter = context['var.current_train_iter']
        train_loss = context['var.train_loss']
        print_strings = [current_time, current_epoch, current_iter, train_loss]
        # add any extra ones
        for extra_train_log in self._extra_train_logs:
            print_strings.append(context[extra_train_log[0]])
        # format the log string with the variable values
        return self._train_format % tuple(print_strings)

    def _post_val_iter_string(self, context):
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        current_epoch = context['var.current_epoch']
        current_iter = context['var.current_val_iter']
        val_loss = context['var.val_loss']
        print_strings = [current_time, current_epoch, current_iter, val_loss]
        for extra_val_log in self._extra_val_logs:
            print_strings.append(context[extra_val_log[0]])

        return self._val_format % tuple(print_strings)


class TQDMConsoleLogger(ConsoleLogger):
    """
    Logs progress using the tqdm progress bar
    """

    def callback_pre_train_epoch(self, context):
        self._pbar = tqdm(total=len(context['component.train_loader']), desc='\r')

    def callback_post_train_iter(self, context):
        self._pbar.update(1)
        self._pbar.set_description('\r' + self._post_train_iter_string(context))

    def callback_post_train_epoch(self, context):
        self._pbar.close()

    def callback_pre_val_epoch(self, context):
        self._pbar = tqdm(total=len(context['component.val_loader']), desc='\r')

    def callback_post_val_iter(self, context):
        self._pbar.update(1)
        self._pbar.set_description('\r' + self._post_val_iter_string(context))

    def callback_post_val_epoch(self, context):
        self._pbar.close()


def _post_epoch_iter_string(context, format_string, extra_logs):
    """
    Utility function to print post-epoch variable values, one per line
    """

    # if user didn't specify any post-epoch logs, just return
    if not len(extra_logs):
        return
    print_strings = []
    for extra_log in extra_logs:
        print_strings.append(context[extra_log[0]])

    return format_string % tuple(print_strings)


class MetricComponent(Component):
    """
    Metric component has the higher default priority than scheduler and monitor
    """

    def __init__(self):
        super().__init__()

class ControllerComponent(Component):
    """
    Scheduler component has low default priority because nothing should depend on it
    """
    def __init__(self):
        super().__init__()


class OnlineAverager(MetricComponent):
    """
    Generic component to average some quantity. Specify the variable in
    the context to average, and the variable name in the context where you
    want to store the average
    """

    def __init__(self, train_in_var=None, val_in_var=None,
                 train_out_var=None, val_out_var=None):
        """
        Args:
            train_in_var: variables to be averaged in training epoch
            val_in_var: variables to be averaged in validation epoch
            train_out_var: output key to store averaged variables in training epoch
            val_out_var: output key to store averaged variables in validation epoch

        Example:
            An Averager component defined as:
            Averager(train_in_var=['var.train_loss'], val_in_var=['var.val_loss'],
                     train_out_var=['var.avg_train_loss'], val_out_var=['var.avg_val_loss'])

            will calculate the average of context['var.train_loss'] in the current training
            epoch after every training iteration, and store the averaged value at context['var.avg_train_loss'].
            Similarly, it will calculate the average of context['var.val_loss'] in the
            current validation epoch after every validation iteration, and store the averaged
            value at context['var.avg_val_loss'].
        """
        super().__init__()
        if train_in_var is not None and train_out_var is None:
            raise ValueError("Specify train_out_var if you specify train_in_var")
        if val_in_var is not None and val_out_var is None:
            raise ValueError("Specify val_out_var if you specify val_in_var")

        if train_in_var is None:
            train_in_var = []
        if val_in_var is None:
            val_in_var = []
        if train_out_var is None:
            train_out_var = []
        if val_out_var is None:
            val_out_var = []

        if not isinstance(train_in_var, list):
            train_in_var = [train_in_var]
        if not isinstance(train_out_var, list):
            train_out_var = [train_out_var]
        if not isinstance(val_in_var, list):
            val_in_var = [val_in_var]
        if not isinstance(val_out_var, list):
            val_out_var = [val_out_var]

        if len(train_out_var) != len(train_in_var):
            raise ValueError("Must provide an output variable for each input variable")
        if len(val_out_var) != len(val_in_var):
            raise ValueError("Must provide an output variable for each input variable")

        self.train_in_var = train_in_var
        self.train_out_var = train_out_var
        self.val_in_var = val_in_var
        self.val_out_var = val_out_var
        self._train_accum = [0 for _ in self.train_in_var]
        self._val_accum = [0 for _ in self.val_in_var]

    def callback_pre_train_epoch(self, context):
        self._train_accum = [0 for _ in self.train_in_var]

    def callback_pre_val_epoch(self, context):
        self._val_accum = [0 for _ in self.val_in_var]

    def _execute_average(self, context, accum, var_in, var_out, cur_iter):
        train_var = context[var_in]

        accum += train_var
        context[var_out] = accum / cur_iter
        return accum

    def callback_post_train_iter(self, context):
        current_iter = context['var.current_train_iter']
        for i, (train_accum, train_in_var, train_out_var) in enumerate(zip(
                self._train_accum, self.train_in_var, self.train_out_var)):
            self._train_accum[i] = self._execute_average(
                context, train_accum, train_in_var,
                train_out_var, current_iter + 1)

    def callback_post_val_iter(self, context):
        current_iter = context['var.current_val_iter']
        for i, (val_accum, val_in_var, val_out_var) in enumerate(zip(
                self._val_accum, self.val_in_var, self.val_out_var)):
            self._val_accum[i] = self._execute_average(
                context, val_accum, val_in_var,
                val_out_var, current_iter + 1)


class OnlineAverageLoss(OnlineAverager):
    """
    Just a conveinence class to averager that has some common defaults for the averager
    """

    def __init__(self, train_in_var='var.train_loss', val_in_var='var.val_loss',
                 train_out_var='var.avg_train_loss', val_out_var='var.avg_val_loss'):
        super().__init__(train_in_var=train_in_var, val_in_var=val_in_var,
                         train_out_var=train_out_var, val_out_var=val_out_var)



class SupervisedTrainer(object):
    """
    This is a basic generic supervised trainer. It will run a training epoch and then
    every 'val_freq', it will run a validation epoch. It expects the following to be
    populated in the context object

    component.train_loader: an iterable that returns a batch of training data,
        in the form of a dict
    component.val_loader: an iterable that returns a batch of validation data,
        in the form of a dict
    component.model: a subclass of torch.nn.module (or a callable class that defines the
        a `train` and `eval` methods), that accepts a batch data_dict as input and
        populates the same data_dict with output fields
    component.loss: a callable that can return a pytorch scalar loss given the data_dict
        returned by component.model
    component.optimizer: a pytorch optimizer that can accept the output of component.loss
        and compute and populate gradients

    The trainer object will set the following variables as it runs

    var.current_epoch
    var.current_train_iter
    var.current_val_iter
    var.total_train_iters: based on the length of the component.train_loader
    var.total_val_iters: based on the length of the component.val_loader
    var.train_loss
    var.val_loss
    var.train_batch_dta
    var.val_batch_data

    If any callback sets the context's var.current_epoch to -1, it will trigger early
    stopping for the training

    """
    def __init__(self, context, val_freq, max_epoch):
        """
        The trainer object needs other components of training.
        """
        # Store the context
        self._context = context
        self._callback_fn_dict = {}

        self.val_freq = val_freq
        self.max_epoch = max_epoch

    def run(self, checkpoint=None):
        """
        Entry point to run a new training

        Args:
            checkpoint: previous context checkpoint. If checkpoint==None, training will run from
                epoch 0.
        """
        self._reset_counter()
        if checkpoint:
            self._context.load(checkpoint)
            self._context.increment('var.current_epoch')
        if self._context.get('component.use_amp') is True:
            self.use_amp = True
            print('---- Using Auto Mixed Precision!')
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_amp = False
        self._run()

    def _reset_counter(self):
        """
        Reset the epoch and iteration counters. Use when start new training.
        """
        self._context['var.current_epoch'] = 0
        self._context['var.current_train_iter'] = 0
        self._context['var.current_val_iter'] = 0

    def _compile_callbacks(self):
        """
        Compile callback function from components
        """
        for key in CALLBACK_FN_NAMES:
            self._callback_fn_dict[key] = collect_callback_fn(self._context, key)

    def _run(self):
        """
        Entry point to run training.
        """
        logging.info('Running trainer with context:\n{}'.format(self._context))
        for epoch in range(self._context['var.current_epoch'], self.max_epoch):
            self._compile_callbacks()
            self._callback_fn_dict['callback_pre_train_epoch'](self._context)
            self._run_train_epoch()
            self._callback_fn_dict['callback_post_train_epoch'](self._context)

            # only run validation every val_freq epochs
            if self.val_freq != 0 and (epoch + 1) % self.val_freq == 0:
                self._callback_fn_dict['callback_pre_val_epoch'](self._context)
                self._run_val_epoch()
                self._callback_fn_dict['callback_post_val_epoch'](self._context)

            # check to make sure no callback triggered early stopping
            if self._context['var.current_epoch'] == -1:
                break
            self._context.increment('var.current_epoch')

    def _run_train_epoch(self):
        """
        Run training for one epoch
        """
        train_loader = self._context['component.train_loader']
        self._context['var.total_train_iters'] = len(train_loader)
        self._context['var.current_train_iter'] = 0

        # set the model to training mode
        self._context['component.model'].train()

        # Training loop with callbacks
        for batch_data in train_loader:

            # set the batch dict to the context
            self._context['var.batch_data'] = batch_data
            # run any logic before the training step
            self._callback_fn_dict['callback_pre_train_iter'](self._context)
            # run the training step
            self._run_train_step()
            # run any logic after the training step
            self._callback_fn_dict['callback_post_train_iter'](self._context)
            # increment training iteration
            self._context.increment('var.current_train_iter')
            # clear the memory since we are finished with it
            self._context.pop('var.batch_data')

    def _run_val_epoch(self):
        """
        Run validation for one epoch
        """
        val_loader = self._context['component.val_loader']
        self._context['var.total_val_iters'] = len(val_loader)
        self._context['var.current_val_iter'] = 0

        # set the model to eval mode
        self._context['component.model'].eval()
        with torch.no_grad():
            # Validation loop with callbacks
            for batch_data in val_loader:
                self._context['var.batch_data'] = batch_data
                self._callback_fn_dict['callback_pre_val_iter'](self._context)
                self._run_val_step()
                self._callback_fn_dict['callback_post_val_iter'](self._context)
                self._context.increment('var.current_val_iter')
                self._context.pop('var.batch_data')

    def _run_train_step(self):
        """
        Run one training step
        """
        # DataParllel only supports dict (not FlatDict)
        # so we use dict for the model input
        batch_data = dict(self._context['var.batch_data'])

        if self.use_amp:
            self._context['component.optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                batch_data = self._context['component.model'](batch_data)
                # update the context
                self._context['var.batch_data'] = batch_data
                # make sure loss and context use the same batch_data ref
                loss = self._context['component.loss'](self._context['var.batch_data'])
            self._context['var.train_loss'] = loss.item()
            self.scaler.scale(loss).backward()
            self.scaler.step(self._context['component.optimizer'])
            self.scaler.update()
        else:

            # Run model forward
            batch_data = self._context['component.model'](batch_data)
            # update the context
            self._context['var.batch_data'] = batch_data
            # Compute loss
            # make sure loss and context use the same batch_data ref
            loss = self._context['component.loss'](self._context['var.batch_data'])
            self._context['var.train_loss'] = loss.item()
            # Perform backprop
            self._context['component.optimizer'].zero_grad()

            loss.backward()
            self._context['component.optimizer'].step()

    def _run_val_step(self):
        """
        Run one validation step
        """
        # DataParllel only supports dict (not FlatDict)
        # so we use dict for the model input
        batch_data = dict(self._context['var.batch_data'])

        if self.use_amp:
            with torch.cuda.amp.autocast():
                batch_data = self._context['component.model'](batch_data)
                # update the context
                self._context['var.batch_data'] = batch_data
                # make sure loss and context use the same batch_data ref
                loss = self._context['component.loss'](self._context['var.batch_data'])
            self._context['var.val_loss'] = loss.item()
        else:
            # Run model forward
            batch_data = self._context['component.model'](batch_data)
            # update the context
            self._context['var.batch_data'] = batch_data
            # Compute loss
            # make sure loss and context use the same batch_data ref
            loss = self._context['component.loss'](self._context['var.batch_data'])
            self._context['var.val_loss'] = loss.item()


class Checkpointer(ControllerComponent):
    """
    Controller that saves checkpoint of the context after training epochs.
    """
    def __init__(self, output_dir, ckpt_freq=1, keep_all_ckpts=True):
        super().__init__()
        self.output_dir = output_dir
        self.ckpt_freq = ckpt_freq
        self.keep_all_ckpts = keep_all_ckpts

    def callback_post_train_epoch(self, context):
        epoch = context['var.current_epoch']
        # only save checkpoint every ckpt_freq epochs
        if (epoch + 1) % self.ckpt_freq == 0:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
            dlt_print('Saving checkpoint of epoch {}'.format(epoch))
            context.save(self.output_dir + '/last_checkpoint.ckpt')
            if self.keep_all_ckpts:
                context.save(self.output_dir + '/epoch_{}.ckpt'.format(epoch))

class OptimScheduler(ControllerComponent):
    """
    Utility class for wrapping pytorch-like schedulers to operate with dlt trainers
    and context
    """

    def __init__(self, scheduler, metric_field=None, after_val=True, epoch_field='var.current_epoch'):
        """
        if after_val is false, callback operates after train epochs,
        metric_field: some schedulers requires a value to operate their logic, in
            this case it's the field name to pull the metric value from
        epoch_field: where to pull the current epoch number from the context
        """
        super().__init__()
        self.scheduler = scheduler
        self.metric_field = metric_field
        self.epoch_field = epoch_field
        self.after_val = after_val

    def _perform_callback(self, context):
        if self.metric_field:
            self.scheduler.step(context[self.metric_field])
        else:
            self.scheduler.step()

    def callback_post_train_epoch(self, context):
        if self.after_val:
            return
        self._perform_callback(context)

    def callback_post_val_epoch(self, context):
        if not self.after_val:
            return
        self._perform_callback(context)
