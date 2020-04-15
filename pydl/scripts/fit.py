import os


def fit(model,
        x, y,
        save_built_model: bool = False,
        model_filename: str = '',
        output_dir=''):

    """
    Fit model

    :param model:
    :param x: input data to predict
    :param y: target values
    :param save_built_model: whether to save the built model in a H5 file or not
    :param model_filename:
    :param output_dir: output directory for the output file

    :return: X and Y used to fit the model
    """

    model.fit(x, y)

    if save_built_model:
        if model_filename == '':
            model_filename = '{}_built.h5'.format(model.name)

        model_filepath = os.path.join(output_dir, model_filename)
        model.save(filepath=model_filepath)

    return x, y
