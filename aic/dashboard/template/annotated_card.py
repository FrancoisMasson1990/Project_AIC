import dash_bootstrap_components as dbc
import dash_core_components as dcc
import plotly.express as px
from dash import html

annotation_colormap = px.colors.qualitative.Light24
annotation_types = ["valve", "calcium", "artefact"]
DEFAULT_ATYPE = annotation_types[0]

# prepare bijective type<->color mapping
typ_col_pairs = [
    (t, annotation_colormap[n % len(annotation_colormap)])
    for n, t in enumerate(annotation_types)
]
# types to colors
color_dict = {}
# colors to types
type_dict = {}
for typ, col in typ_col_pairs:
    color_dict[typ] = col
    type_dict[col] = typ

options = list(color_dict.keys())


def annotated_data_card():
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H4("Create new annotation for"),
                                dcc.Dropdown(
                                    id="annotation-type-dropdown",
                                    options=[
                                        {"label": t, "value": t}
                                        for t in annotation_types
                                    ],
                                    value=DEFAULT_ATYPE,
                                    clearable=False,
                                ),
                            ],
                            align="center",
                        )
                    ),
                ]
            ),
        ],
        color="#1E1E1E",
        style={
            "marginLeft": "10px",
            "border-style": "dashed",
            "border-color": "#F8F2F2",
            "border-radius": "5px",
            "border-width": "1px",
        },
    )
