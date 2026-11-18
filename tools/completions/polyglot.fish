# Fish completion for polyglot
complete -c polyglot -f

complete -c polyglot -n "__fish_use_subcommand" -a "run" -d "Compile and execute a file"
complete -c polyglot -n "__fish_use_subcommand" -a "build" -d "Compile a file"
complete -c polyglot -n "__fish_use_subcommand" -a "check" -d "Parse, translate, and check termination"
complete -c polyglot -n "__fish_use_subcommand" -a "lsp" -d "Run the language server over stdio"
complete -c polyglot -n "__fish_use_subcommand" -a "exec" -d "Compile and execute with machine-readable metrics"
complete -c polyglot -n "__fish_use_subcommand" -a "visualizer" -d "Serve the interactive AST/monomorphization visualizer"
complete -c polyglot -n "__fish_use_subcommand" -a "gui" -d "Alias for visualizer"
complete -c polyglot -n "__fish_use_subcommand" -a "compile" -d "Compatibility compile command"

complete -c polyglot -n "__fish_seen_subcommand_from run" -a "(__fish_complete_path)" -d "Polyglot file"
complete -c polyglot -n "__fish_seen_subcommand_from run" -l backend -a "python native" -d "Backend"
complete -c polyglot -n "__fish_seen_subcommand_from run" -l target-width -d "Target width (32 or 64)"
complete -c polyglot -n "__fish_seen_subcommand_from run" -l verbose -d "Verbose errors"

complete -c polyglot -n "__fish_seen_subcommand_from build" -a "(__fish_complete_path)" -d "Polyglot file"
complete -c polyglot -n "__fish_seen_subcommand_from build" -l backend -a "python native" -d "Backend"
complete -c polyglot -n "__fish_seen_subcommand_from build" -l out-dir -d "Output directory"
complete -c polyglot -n "__fish_seen_subcommand_from build" -l verbose -d "Verbose errors"

complete -c polyglot -n "__fish_seen_subcommand_from check" -a "(__fish_complete_path)" -d "Polyglot file"
complete -c polyglot -n "__fish_seen_subcommand_from check" -l verbose -d "Verbose errors"

complete -c polyglot -n "__fish_seen_subcommand_from exec" -a "(__fish_complete_path)" -d "Polyglot file"
complete -c polyglot -n "__fish_seen_subcommand_from exec" -l stdin -d "Read from stdin"
complete -c polyglot -n "__fish_seen_subcommand_from exec" -l backend -a "python native wasm jit" -d "Backend"
complete -c polyglot -n "__fish_seen_subcommand_from exec" -l target-width -d "Target width (32 or 64)"
complete -c polyglot -n "__fish_seen_subcommand_from exec" -l result-file -d "Write JSON metadata"
complete -c polyglot -n "__fish_seen_subcommand_from exec" -l verbose -d "Verbose errors"

complete -c polyglot -n "__fish_seen_subcommand_from visualizer gui" -l host -d "Interface to bind"
complete -c polyglot -n "__fish_seen_subcommand_from visualizer gui" -l port -d "Port to bind"
complete -c polyglot -n "__fish_seen_subcommand_from visualizer gui" -l no-open -d "Do not open a browser"
complete -c polyglot -n "__fish_seen_subcommand_from visualizer gui" -l verbose -d "Log HTTP requests"

complete -c polyglot -n "__fish_seen_subcommand_from compile" -l backend -a "python native" -d "Backend"
complete -c polyglot -n "__fish_seen_subcommand_from compile" -l stdin -d "Read from stdin"
complete -c polyglot -n "__fish_seen_subcommand_from compile" -l file -d "Input file"
complete -c polyglot -n "__fish_seen_subcommand_from compile" -l trusted -d "Trusted mode"
complete -c polyglot -n "__fish_seen_subcommand_from compile" -l verbose -d "Verbose errors"
