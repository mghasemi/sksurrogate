/**
 * Which dataset partition a request should read when the user has not chosen one.
 *
 * The server resolves an *omitted* partition to `train` when it exists and
 * otherwise to the task's first registered partition (see `api/partitions.py`).
 * That default only applies when the field is absent: a frontend that hard-codes
 * `"train"` turns every omission into an explicit name, which 404s for tasks
 * stored under another partition (e.g. a single `main`, or one named after the
 * task). So the pages derive the same default from the dataset metadata and send
 * the name only when it is known.
 */

import { useQuery } from "@tanstack/react-query";

import { ApiError, getDatasetMetadata } from "../api/client";

/** The registered partition to read: `train` when it exists, else the first name. */
export function defaultPartition(
  partitions: Record<string, unknown> | undefined | null,
): string {
  const names = Object.keys(partitions ?? {});
  if (names.length === 0) return "";
  if (names.includes("train")) return "train";
  return [...names].sort()[0];
}

/**
 * The task's default partition name, or `""` until the metadata resolves.
 *
 * `""` is deliberate: callers treat it as "not named yet" and omit the field, so
 * the server's own resolution applies instead of a guessed name.
 */
export function useDefaultPartition(task: string | null | undefined): string {
  const metaQ = useQuery({
    queryKey: ["dataset-partition-default", task],
    queryFn: async () => {
      try {
        return await getDatasetMetadata(task as string);
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return null;
        throw err;
      }
    },
    enabled: !!task,
    retry: false,
    staleTime: 60_000,
  });
  return defaultPartition(metaQ.data?.partitions);
}
