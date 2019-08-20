from enum import Enum
import math
from typing import Dict, Any

# abstraction prefix
COL_PREFIX = 'col#'
VAL_PREFIX = 'val#'


class Bind(Enum):
    Table = "tab"
    # all value of date is always date, not value self
    ValueText = "val_text"
    ValueDate = "value_date"
    ValueNumber = "value_number"
    ColumnText = "col_text"
    ColumnDate = "col_date"
    ColumnNumber = "col_number"

    # abstract tag, used for simplify
    Column = "[ABSTRACT]Column"
    Value = "[ABSTRACT]Value"
    Deter = "[ABSTRACT]Deter"

    AllDeter = "all"
    OtherDeter = "other"
    NextDeter = "next"
    PreDeter = "pre"
    ThatDeter = "that"

    Item = "item"
    ItemPossess = "item_possess"
    PersonPossess = "person_possess"
    ItemEntity = "item_entity"

    # These bind types are kept for symbol matching
    ComOp = "comparator"
    AggOp = "aggregation"
    Exclude = "excluding"
    Dir = "direction"


COLUMN_BIND_TYPES = [Bind.ColumnText, Bind.ColumnDate, Bind.ColumnNumber]

VALUE_BIND_TYPES = [Bind.ValueText, Bind.ValueDate, Bind.ValueNumber]

DETER_BIND_TYPES = [Bind.PreDeter, Bind.NextDeter, Bind.ThatDeter, Bind.AllDeter, Bind.OtherDeter]

PRONOUN_BIND_TYPES = [Bind.PreDeter, Bind.NextDeter, Bind.ThatDeter, Bind.AllDeter,
                      Bind.ItemPossess, Bind.Item, Bind.ItemEntity, Bind.PersonPossess]


class StandardSymbol(object):
    def __init__(self, origin, header, class_type):
        self.origin = origin
        self.header = header
        self.class_type = class_type

    def __eq__(self, other):
        if other is None or not other.class_type == self.class_type:
            return False
        if self.class_type in VALUE_BIND_TYPES:
            if self.origin == other.origin:
                return True
            else:
                return False
        elif self.class_type in COLUMN_BIND_TYPES:
            if self.header == other.header:
                return True
            else:
                return False
        elif self.class_type != Bind.Exclude:
            if self.origin == other.origin:
                return True
            else:
                return False
        else:
            return True

    def __hash__(self):
        if self.class_type == Bind.ValueText:
            return hash((self.origin, self.class_type.name))
        elif self.class_type == Bind.ValueDate or self.class_type == Bind.ValueNumber:
            return hash((self.origin, self.class_type.name))
        elif self.class_type == Bind.ColumnText or self.class_type == Bind.ColumnDate or self.class_type == Bind.ColumnNumber:
            return hash((self.header, self.class_type.name))
        elif self.class_type == Bind.Exclude:
            return hash(self.class_type.name)
        else:
            return hash(self.origin)


