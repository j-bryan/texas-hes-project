import sys
from scipy.io import loadmat
import pandas as pd

sys.path.insert(0, 'C:\\Program Files\\Dymola 2022\\Modelica\\Library\\python_interface\\dymola.egg')  # TODO: You might need to change this to match the system you're on
from dymola.dymola_interface import DymolaInterface


def parse_mat(pth):
    """ Parses the .mat files that Dymola uses to store simulation results.
    
    Parameters
    ----------

    """
    mat = loadmat(pth)

    names = [''] * len(mat['name'][0])
    for s in mat['name']:
        for i in range(len(names)):
            names[i] += s[i]
    for i in range(len(names)):
        names[i] = names[i].rstrip()

    # descriptions = [''] * len(mat['description'][0])
    # for s in mat['description']:
    #     for i in range(len(descriptions)):
    #         descriptions[i] += s[i]
    # for i in range(len(descriptions)):
    #     descriptions[i] = descriptions[i].rstrip()
    
    data_info = mat['dataInfo'].T

    params = {}
    data = pd.DataFrame()

    for name, info in zip(names, data_info):
        table_num, index, _, _ = info
        if table_num == 1:
            params[name] = mat['data_1'][index - 1][0]
        elif table_num == 2:
            data[name] = mat['data_2'][index - 1]
        else:  # Only happens for 'Time'
            params['t_start'] = mat['data_1'][index - 1][0]
            params['t_stop'] = mat['data_1'][index - 1][1]
            data[name] = mat['data_2'][index - 1]

    return data, params


def generate_simulator(pkg_pth, model, design_params, output_pth):
    """ Generates a function which takes a list-like input of design parameters and outputs the simulation results and parameters.
    
    Parameters
    ----------
    pkg_pth: str
        Path to the directory with the Dymola package containing the model to simulate
    model: str
        Modelica-style name of the model within package to simulate
    design_params: array_like, size (N:)
        Names of parameters to be set in the Dymola model
    output_pth: str
        Path to a directory plus file name (without extension!) where Dymola will store the simulation results in a .mat file

    Returns
    -------
    f: callable
        The generated simulation wrapper function
    dymola: dymola.dymola_interface.DymolaInterface
        The dymola interface instance created to run the simulations. This should be closed before exiting the script with the .close() method.
    """

    # Open the Dymola interface, open the specified model, and translate the model.
    dymola = DymolaInterface()
    dymola.openModel(path=pkg_pth)
    dymola.translateModel(model)

    # Generate the simulation wrapper
    def f(x):
        """ Dymola simulation wrapper
        
        Parameters
        ----------
        x: array_like, size (N:) (i.e. matches the length of design_params specified in the outer function)
            Array of parameter values to set. Must be able to be cast to a list if it is not already of type list.

        Returns
        -------
        data: pandas.DataFrame
            The simulation data. The time series values of a variable can be accessed by its Dymola name (i.e. data['<Variable.Name.Here>'])
        params: dict
            Simulation parameters
        """
        if not isinstance(x, list): x = list(x)  # x needs to be a list. This line allows scipy functions to pass numpy arrays also.

        # Run the simulation. The included parameters here seem to work well for our HES model.
        dymola.simulateExtendedModel(
            problem=model,
            startTime=0.0,
            stopTime=3600*24*366,
            outputInterval=3600,
            method='Lsodar',
            tolerance=1e-3,
            resultFile=output_pth,
            initialNames=design_params,
            initialValues=x
        )

        # Parse the resulting .mat file to extract the simulation data
        data, params = parse_mat(output_pth + '.mat')
        
        return data, params
    
    return f, dymola
