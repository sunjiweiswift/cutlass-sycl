#!/bin/bash

message="$1"
color="$2"

print_message() {
  local message="$1"
  local color_name="$2"
  local color_code

  # Return immediately if message is empty
  if [[ -z "$message" ]]; then
    return
  fi

  case $color_name in
    black)
      color_code="\033[0;30m"
      ;;
    red)
      color_code="\033[0;31m"
      ;;
    green)
      color_code="\033[0;32m"
      ;;
    yellow)
      color_code="\033[0;33m"
      ;;
    blue)
      color_code="\033[0;34m"
      ;;
    magenta)
      color_code="\033[0;35m"
      ;;
    cyan)
      color_code="\033[0;36m"
      ;;
    white)
      color_code="\033[0;37m"
      ;;
    bright_black)
      color_code="\033[1;30m"
      ;;
    bright_red)
      color_code="\033[1;31m"
      ;;
    bright_green)
      color_code="\033[1;32m"
      ;;
    bright_yellow)
      color_code="\033[1;33m"
      ;;
    bright_blue)
      color_code="\033[1;34m"
      ;;
    bright_magenta)
      color_code="\033[1;35m"
      ;;
    bright_cyan)
      color_code="\033[1;36m"
      ;;
    bright_white)
      color_code="\033[1;37m"
      ;;
    *)
      color_code="\033[0m" # Default to no color
      ;;
  esac

  local length=${#message}
  local separator
  separator="$(printf '=%.0s' $(seq 1 "$length"))"

  echo -e "${color_code}${separator}\033[0m"
  echo -e "${color_code}${message}\033[0m"
  echo -e "${color_code}${separator}\033[0m"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_message "$message" "$color"
fi