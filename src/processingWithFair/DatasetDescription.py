class DatasetDescription():

    def __init__(self, result_path, orig_path, protected_attribute, protected_group,
                 header, judgment, alpha=0.1):
        self.__result_path = result_path
        self.__orig_data_path = orig_path
        self.__protected_attribute = protected_attribute
        self.__protected_group = protected_group
        self.__header = header
        self.__judgment = judgment
        self.__alpha = alpha

    @property
    def orig_data_path(self):
        return self.__orig_data_path

    @property
    def result_path(self):
        return self.__result_path

    @property
    def protected_attribute(self):
        return self.__protected_attribute

    @property
    def protected_group(self):
        return self.__protected_group

    @property
    def header(self):
        return self.__header

    @property
    def judgment(self):
        return self.__judgment

    @property
    def alpha(self):
        return self.__alpha
