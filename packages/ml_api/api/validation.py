from marshmallow import Schema, fields, ValidationError
import typing as t



class InvalidInputError(Exception):
    """Invalid model input."""


class TitanicDataRequestSchema(Schema):
    pclass = fields.Integer()
    sex = fields.String()
    age = fields.Integer()
    sibsp = fields.Integer()
    parch = fields.Integer() 
    fare = fields.Float()
    cabin = fields.String()
    embarked = fields.String()

    name = fields.String(allow_none=True)
    ticket = fields.String(allow_none=True)
    boat = fields.String(allow_none=True)
    body = fields.String(allow_none=True)
    home_dest = fields.String(allow_none=True)


def _filter_error_rows(errors: dict, validated_data: t.List[dict]):
    indexes = errors.keys()

    for index in sorted(indexes, reverse=True):
        del validated_data[index]

    return validated_data


def validate_inputs(inputs):
    """Check prediction inputs against schema"""

    schema = TitanicDataRequestSchema(many=True, strict=True)

    errors = None
    try:
        schema.load(inputs)
    except ValidationError as err:
        errors = err.messages 

    if errors:
       validated_inputs =  _filter_error_rows(errors=errors, validated_data=inputs)
    else:
        validated_inputs = inputs
    
    return validated_inputs, errors

