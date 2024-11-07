#!/bin/bash

python -m src.api.server &
python -m src.bot
wait