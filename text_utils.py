def load_data(filename:str) -> list[str]:
    """
    Load data from text file and convert it into a list of strings.

    Parameters
    ----------
    filename : str
        Path to a UTF-8 encoded text file. Each line is treated as a separate entry.

    Returns
    -------
    data_list : list of str
        List where each element is a line from the file with leading/trailing whitespace removed.
    
    Examples
    -------
    X_train = load_data('x_train.txt')
    y_train = load_data('y_train.txt')
    """

    data_list = []
    with open(filename, encoding='utf-8') as fhand:
        for line in fhand:
            data_list.append(line.strip())
    return data_list


