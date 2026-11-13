_polyglot_completions()
{
    local cur prev words cword
    _init_completion || return

    if [[ $cword -eq 1 ]]; then
        COMPREPLY=($(compgen -W "run build check compile" -- "$cur"))
        return
    fi

    case "${words[1]}" in
        run|build|check)
            COMPREPLY=($(compgen -f -- "$cur"))
            ;;
        compile)
            COMPREPLY=($(compgen -W "--backend --stdin --file --trusted --verbose" -- "$cur"))
            ;;
    esac
}
complete -F _polyglot_completions polyglot
