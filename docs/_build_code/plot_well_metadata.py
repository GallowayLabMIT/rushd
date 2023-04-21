# ruff: noqa
from pathlib import Path
import rushd as rd

# Monkeypatch nonsense to make the included code easy to read
def passthrough(func):
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner


original_func = passthrough(rd.plot.plot_well_metadata)


def wrapper(func, path):
    def inner(*args, **kwargs):
        arg_list = list(args)
        arg_list[0] = Path("../_static") / args[0]
        kwargs["output_dir"] = path
        return func(*arg_list, **kwargs)

    return inner


rd.plot.plot_well_metadata = wrapper(
    original_func, Path("../_static/built_output/plot_metadata_overall")
)

import rushd as rd

rd.plot.plot_well_metadata("example_plot_metadata.yaml")

rd.plot.plot_well_metadata = wrapper(
    original_func, Path("../_static/built_output/plot_metadata_custom")
)

rd.plot.plot_well_metadata("example_plot_metadata.yaml", columns=["dox"], style={"dox": "linear"})
rd.plot.plot_well_metadata(
    "example_plot_metadata.yaml",
    columns=["orientation"],
    style={"orientation": {"tandem": "#19D2BF", "convergent": "#FFB133", "divergent": "#FE484E"}},
)
rd.plot.plot_well_metadata("example_6well_plot_metadata.yaml", plate_size=(3, 2))
