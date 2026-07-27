# RoboCOIN Missing Language-Action Fallback

## Goal

Keep RoboCOIN episodes in numerical action training when their optional
`language_action` field or text file is absent. Such samples must use the
standard VLM message construction instead of terminating a distributed run.

## Behavior

- If a non-empty language action is available in the sample or its configured
  text file, construct VLM inputs with `preprocess_vlm_messages_lap`.
- If neither source exists, set the per-sample language action to `None` and
  construct VLM inputs with `preprocess_vlm_messages`.
- Preserve `initial_state`, `action_sequence`, canonical55 mapping, and numeric
  action loss for fallback samples.
- Only missing optional language-action data triggers fallback. Existing but
  unreadable files, empty files, invalid episode metadata, and other data
  integrity failures remain errors.

## Scope

The change is limited to `LeRobotRoboCOINDataset`. Other dataset loaders retain
their current language-action requirements and error behavior.

## Implementation Shape

The language-action loader will return `None` only when metadata does not
declare a language-action path and the conventional per-episode file does not
exist. The sample construction branch will choose LAP preprocessing only for a
non-`None` language action; otherwise it will use standard VLM preprocessing.

## Verification

Add focused tests proving:

1. A missing conventional language-action file returns `None` and selects
   standard VLM preprocessing.
2. An available language action continues to select LAP preprocessing.
3. A metadata-declared but missing/unreadable path still raises an error.
4. Numerical state and action outputs are unchanged by the fallback.

