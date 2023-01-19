import {GridWorldTask} from "../gridworld/gridworld-task.js"
import {errors} from "../exputils.js"

jsPsych.plugins["GridNavigation"] = (function() {
    var plugin = {};
    plugin.info = {
        name: 'GridNavigation',
        parameters: {}
    };
        plugin.trial = errors.handle(async function(display_element, trial) {
            let trialparams = Object.assign({
                navigationText: "&nbsp;",
                participantStarts: true,
                MAX_TIMESTEPS: 1e10,
                TILE_SIZE: 50,
                OBJECT_ANIMATION_TIME: 50,

                showPoints: true,
                initialPoints: 100,
                dollarsPerPoint: .2/100,
                goalCountdown: false,
                hideObstaclesOnMove: false,
                hideBrokenPiecesOnMove: false,
                GOALSIZE: .4,
                INITIALGOAL_COUNTDOWN_MS: 5000,
                GOAL_COUNTDOWN_MS: 1000
            }, trial.trialparams)
            let taskparams = trial.taskparams;
            let trialData = {
                trialparams,
                taskparams,
                sessionId: window.globalExpStore.sessionId,
                navigationData: [],
                datatype: "trialData"
            }
            window.trialData = trialData;

            const startTime = Date.now();
            display_element.innerHTML = '<div id="messages"></div><div id="points"></div><div id="gridworld"></div>';
            let trialnum = 0;
            let goalCountdown;
            let points = trialparams.initialPoints;
            if (trialparams.showPoints) {
                $("#points").html(`Points: ${points}`)
            }

            let task = new GridWorldTask({
                container: $("#gridworld")[0],
                step_callback: (d) => {
                    if (trialnum === 0) {
                        $("#messages").html(trialparams.navigationText);
                    }
                    if (trialparams.hideBrokenPiecesOnMove) {
                        if (trialnum == 0) {
                            hideBrokenPieces(task, taskparams.feature_colors);
                        }
                    }
                    if (trialparams.hideObstaclesOnMove) {
                        if (trialnum == 0) {
                            hideObstacles(task);
                        }
                    }
                    if (typeof(goalCountdown) !== "undefined") {
                        goalCountdown.reset()
                        if (goalCountdown.bonusTime) {
                            points = points + d.reward;
                            points = points < 0 ? 0 : points;
                        }
                        else {
                            points = 0
                        }
                    }
                    if (trialparams.showPoints) {
                        $("#points").html(`Points: ${points}`)
                    }
                    d.trialnum = trialnum;
                    d.sessionId = window.globalExpStore.sessionId;
                    trialData.navigationData.push(d)
                    trialnum += 1;
                    if (trialnum >= trialparams.MAX_TIMESTEPS) {
                        task.end_task()
                    }
                },
                endtask_callback: () => {
                    setTimeout(function() {
                        window.globalExpStore.bonusDollars += points*trialparams.dollarsPerPoint;
                        display_element.innerHTML = '';
                        jsPsych.finishTrial({data: trialData})
                    }, trialparams.ENDTASK_TIMEOUT || 0);
                },
                TILE_SIZE: trialparams.TILE_SIZE,
                OBJECT_ANIMATION_TIME: trialparams.OBJECT_ANIMATION_TIME
            });

            if (trialparams.participantStarts) {
                await participantRevealsMaze(task)
            }
            $("#messages").html(trialparams.navigationText);
            task.init(taskparams);
            task.start()
            if (trialparams.goalCountdown) {
                goalCountdown = createGoalCountdown(task, trialparams);
                goalCountdown.init()
            }

            // methods
            let hideObstacles = (task) => {
                task.painter.tiles.filter((t) => {
                    let s = t['grid_xy'];
                    return !['#', '.', '$'].includes(task.mdp.location_features[s])
                }).map((t) => {
                    t.attr('fill', 'white')
                })
            }
            let hideBrokenPieces = (task, feature_colors) => {
                for (let i = 0; i < task.painter.tiles.length; i++) {
                    let tile = task.painter.tiles[i];
                    let xy = tile.grid_xy;
                    let feature = task.mdp.location_features[xy];
                    if (['a', 'b', 'c', 'd', 'e', 'f'].includes(feature)) {
                        let upperColor = feature_colors[feature.toUpperCase()];
                        tile.attr('fill', upperColor)
                    }
                }
            }
            //
            // let task = new GridWorldTask({
            //     container: taskdiv,
            //     step_callback: (d) => {
            //         if (trial.goalCountdown){
            //             resetGoal(task.painter);
            //         }
            //         if (trial.hideObstaclesOnMove) {
            //             if (trialnum == 0) {
            //                 hideObstacles(task);
            //             }
            //         }
            function createGoalCountdown(task, trialparams) {
                let offset = (1 - trialparams.GOALSIZE)/2;
                let goalloc = task.mdp.absorbing_locations[0].split(",").map(Number);
                let painter = task.painter;
                let fullGoalParams = {
                    x : (goalloc[0]+offset)*painter.TILE_SIZE+painter.DISPLAY_BORDER,
                    y : painter.y_to_h(goalloc[1] - offset)*painter.TILE_SIZE+painter.DISPLAY_BORDER,
                    width : painter.TILE_SIZE*trialparams.GOALSIZE,
                    height : painter.TILE_SIZE*trialparams.GOALSIZE,
                }
                let emptyGoalParams = {
                    x: (goalloc[0]+.5)*painter.TILE_SIZE+painter.DISPLAY_BORDER,
                    y: painter.y_to_h(goalloc[1] - .5)*painter.TILE_SIZE+painter.DISPLAY_BORDER,
                    width : 0,
                    height : 0,
                }
                let gc = {
                    bonusTime: true,
                    init: () => {
                        gc.goalobj = painter.paper.add([
                            Object.assign(
                            {
                                type: "rect",
                                fill : 'green',
                                stroke: 'white',
                                "stroke-width": 1
                            },
                            fullGoalParams
                            )
                        ]);
                        gc.goalanim = gc.goalobj.animate(
                            emptyGoalParams,
                            trialparams.INITIALGOAL_COUNTDOWN_MS,
                            "linear",
                            () => {gc.bonusTime = false}
                        )
                    },
                    reset : () => {
                        if (!gc.bonusTime) {
                            return
                        }
                        gc.goalanim.stop();
                        gc.goalobj.attr(fullGoalParams)
                        gc.goalanim = gc.goalobj.animate(
                            emptyGoalParams,
                            trialparams.GOAL_COUNTDOWN_MS,
                            "linear",
                            () => {gc.bonusTime = false}
                        )
                    }
                };
                return gc
            }

            async function participantRevealsMaze(task) {
                $("#messages").html("Press <u>space</u> to begin the round.");
                task.init({
                    tile_array: removeNonWalls(taskparams['tile_array'])
                })
                return new Promise((resolve) => {
                    $(document).on("keydown.start_task", (e) => {
                        let kc = e.keyCode ? e.keyCode : e.which;
                        if (kc !== 32) {
                            return
                        }
                        e.preventDefault();
                        $(document).off("keydown.start_task");
                        resolve();
                    });
                })
            }

            function removeNonWalls(task_array) {
                let ignoreFeatures = [
                    'a', 'A',
                    'b', 'B',
                    'c', 'C',
                    'd', 'D',
                    'e', 'E',
                    '@', '$',
                    's', 'g'
                ];
                let new_array = []
                let row;
                for (var y = 0; y < task_array.length; y++) {
                    row = [];
                    for (var x = 0; x < task_array[0].length; x++) {
                        if (ignoreFeatures.includes(task_array[y][x])) {
                            row.push('.')
                        }
                        else {
                            row.push(task_array[y][x])
                        }
                    }
                    new_array.push(row.join(''))
                }
                return new_array
            }






            //
            // console.log(trial);
            // const startTime = Date.now();
            // display_element.innerHTML = '<div id="messages"></div><div id="gridworld"></div>';
            // let taskdiv = $("#gridworld")[0];
            // let msgdiv = $("#messages")[0];
            // let trialnum = 0;
            // let addToBonus = typeof(trial.bonus) !== "undefined" ? trial.bonus : true;
            // trial.goalCountdown = typeof(trial.goalCountdown) === "undefined" ? false : trial.goalCountdown;
            // trial.hideObstaclesOnMove = typeof(trial.hideObstaclesOnMove) === "undefined" ? false : trial.hideObstaclesOnMove;
            // trial.message = trial.message || "&nbsp;";
            // trial.INITIALGOAL_COUNTDOWN_SEC = trial.INITIALGOAL_COUNTDOWN_SEC || 5000;
            // trial.GOAL_COUNTDOWN_SEC = trial.GOAL_COUNTDOWN_SEC || 1000;
            // trial.GOALSIZE = trial.GOALSIZE || .4;
            // trial.TILE_SIZE = trial.TILE_SIZE || 50
            // trial.OBJECT_ANIMATION_TIME = trial.OBJECT_ANIMATION_TIME || 50
            // let trialData = [];
            //
            //
            // //hiding obstacles
            // let hideObstacles = (task) => {
            //     task.painter.tiles.filter((t) => {
            //         let s = t['grid_xy'];
            //         return ['0', '1', '2', '3', '4', '5', '6'].includes(task.mdp.state_features[s])
            //     }).map((t) => {
            //         t.attr('fill', 'white')
            //     })
            // }
            //
            // let task = new GridWorldTask({
            //     container: taskdiv,
            //     step_callback: (d) => {
            //         if (trial.goalCountdown){
            //             resetGoal(task.painter);
            //         }
            //         if (trial.hideObstaclesOnMove) {
            //             if (trialnum == 0) {
            //                 hideObstacles(task);
            //             }
            //         }
            //         d.type = trial.type;
            //         d.round = trial.round;
            //         d.roundtype = trial.roundtype;
            //         d.gridname = trial.taskparams.name;
            //         d.trialnum = trialnum;
            //         d.sessionId = window.globalExpStore.sessionId;
            //         trialnum += 1;
            //         trialData.push(d);
            //     },
            //     endtask_callback: () => {
            //         setTimeout(function() {
            //             if (addToBonus && bonusTime) {
            //                 window.globalExpStore.bonusDollars += window.globalExpStore.roundBonusCents/100;
            //                 console.log("bonus!")
            //             }
            //             else {
            //                 console.log("no bonus :(")
            //             }
            //             display_element.innerHTML = '';
            //             jsPsych.finishTrial({data: trialData})
            //         }, trial.ENDTASK_TIMEOUT || 0);
            //     },
            //     TILE_SIZE: trial.TILE_SIZE,
            //     OBJECT_ANIMATION_TIME: trial.OBJECT_ANIMATION_TIME
            // });
            //
            // trial.participantStarts = typeof(trial.participantStarts) === 'undefined' ? true : trial.participantStarts
            // window.task = task;
            // if (trial.participantStarts) {
            //     $(msgdiv).html("Press <u>space</u> to begin the round.");
            //     let gridheight = trial.taskparams['feature_array'].length;
            //     let gridwidth = trial.taskparams['feature_array'][0].length;
            //     let row = '.'.repeat(gridwidth);
            //     trial.emptygrid = trial.emptygrid || Array(gridheight).fill(row);
            //     task.init({feature_array: trial.emptygrid});
            //
            //     //press space to start task
            //     $(document).on("keydown.start_task", (e) => {
            //         let kc = e.keyCode ? e.keyCode : e.which;
            //         if (kc !== 32) {
            //             return
            //         }
            //         e.preventDefault();
            //         $(document).off("keydown.start_task");
            //         $(msgdiv).html(trial.message);
            //         task.init(trial.taskparams);
            //         task.start();
            //         if (trial.goalCountdown){
            //             initGoal(task.painter);
            //         }
            //     });
            // }
            // else {
            //     task.init(trial.taskparams);
            //     task.start();
            //     if (trial.goalCountdown){
            //         initGoal(task.painter);
            //     }
            // }
        });

    return plugin;
})();
