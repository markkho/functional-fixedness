# Psiturk App

Local development:
```
$ make dev
```

## Pushing to heroku
From root:
```
git push heroku `git subtree split --prefix psiturkapp/ master`:refs/heads/master -f
```

## Config format
```json
{
    "params": {
        "cutoff_time_min": 30,
        "expectedTime": "15 minutes",
        "recruitment_platform": "prolific",
        "EXPERIMENT_CODE_VERSION": "ABCDE",
    },
    "preloadImages" : ["static/.../img.png", "..."],
    "timelines" : ["..."]
}
```
