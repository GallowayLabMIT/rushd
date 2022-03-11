import rushd
from rushd.flow import load_with_metadata

yaml_path = "/Users/kaseylove/Desktop/MIT/export_test/test.yaml"
data_path = "/Users/kaseylove/Desktop/MIT/export_test/export"
df = load_with_metadata(data_path, yaml_path)
print(df)

regex = r"^.*export_(?P<well>[A-G0-9]+)_(?P<population>.+)\.csv"
df = load_with_metadata(data_path, yaml_path, filename_regex=regex)
print(df)
